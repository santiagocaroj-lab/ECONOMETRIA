# Importación de librerías necesarias
import streamlit as st # Framework para crear la aplicación web de manera interactiva
import pandas as pd # Para el manejo, indexación y transformación de datos (DataFrames)
import numpy as np # Para operaciones matriciales y manejo especializado de nulos (NaN)
import plotly.express as px # Para la visualización interactiva de gráficos dinámicos
import seaborn as sns # Para la generación de la matriz de calor (Heatmap)
import matplotlib.pyplot as plt # Soporte estructural para gráficos estáticos de Seaborn
import statsmodels.api as sm # Para agregar la constante econométrica al set de regresores
from linearmodels.panel import PanelOLS # Para de panel con doble efecto fijo
import docx # Para compilar y exportar reportes automatizados en formato Word
from docx.shared import Inches # Ajuste de márgenes y dimensiones en el reporte exportable
import io # Para la gestión eficiente de buffers de datos binarios en la memoria virtual

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA INTERFAZ DE USUARIO
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Análisis Econométrico de Panel", layout="wide")
st.title("📊 Análisis Econométrico: Libertad de Prensa, Corrupción e IED")
st.markdown("""
Esta aplicación permite transformar una base de datos en formato ancho a largo, 
explorar descriptivamente las interacciones y estimar modelos dinámicos de panel con controles bidimensionales.
""")

# -----------------------------------------------------------------------------
# BARRA LATERAL: CARGA DE ARCHIVOS DE ENTRADA
# -----------------------------------------------------------------------------
st.sidebar.header("1. Carga de Archivos")
uploaded_data = st.sidebar.file_uploader("Sube tu base de datos (Excel o CSV)", type=["xlsx", "csv"])
uploaded_doc = st.sidebar.file_uploader("Sube el documento Word metodológico (Opcional)", type=["docx"])

# Visualizador del documento metodológico de soporte en la barra lateral
if uploaded_doc is not None:
    st.sidebar.success("Documento Word cargado correctamente.")
    with st.expander("📄 Ver Documento Metodológico"):
        doc = docx.Document(uploaded_doc)
        for para in doc.paragraphs:
            st.write(para.text)

# -----------------------------------------------------------------------------
# MOTOR DE PROCESAMIENTO ECONOMÉTRICO
# -----------------------------------------------------------------------------
if uploaded_data is not None:
    # Lectura del archivo de datos tolerante al formato origen
    try:
        if uploaded_data.name.endswith('.csv'):
            df_wide = pd.read_csv(uploaded_data)
        else:
            df_wide = pd.read_excel(uploaded_data)
        st.success("Base de datos cargada exitosamente.")
    except Exception as e:
        st.error(f"Error crítico al leer el archivo de entrada: {e}")
        st.stop()

    st.header("2. Transformación Estructural de Datos")
    
    # Definición de una lista explícita de stubnames para máxima robustez
    stubnames_lista = ['CPI_SCORE', 'CPI_RANK', 'CPI_RANKING', 'CORRUPCION', 'RSF_SCORE', 'RSF_RANK', 'RSF_RANKING', 'IED_PIB']
    
    # Reestructuración limpia de formato Ancho (Wide) a Largo (Long)
    df_long = pd.wide_to_long(
        df_wide, 
        stubnames=stubnames_lista,
        i='País',                 # Unidad de corte transversal (Individuo)
        j='Año',                  # Unidad temporal (Tiempo)
        sep='_',                  # Carácter delimitador antes del año numérico
        suffix=r'\d+'             # Expresión regular que captura el sufijo entero del año
    ).reset_index()               

    # Ordenamiento jerárquico indispensable para el correcto cálculo de rezagos temporales
    df_long = df_long.sort_values(by=['País', 'Año'])

    # -------------------------------------------------------------------------
    # GENERACIÓN DE VARIABLES REZAGADAS (LAGGED VARIABLES)
    # -------------------------------------------------------------------------
    # Se agrupa estrictamente por país para que los límites nacionales impidan contaminación de datos
    df_long['RSF_L1'] = df_long.groupby('País')['RSF_SCORE'].shift(1)
    df_long['CORRUPCION_L1'] = df_long.groupby('País')['CORRUPCION'].shift(1)
    
    # Verificación defensiva y cálculo del rezago de la Inversión Extranjera Directa
    if 'IED_PIB' in df_long.columns:
        df_long['IED_L1'] = df_long.groupby('País')['IED_PIB'].shift(1)
    else:
        df_long['IED_L1'] = np.nan

    st.write("Vista previa de los datos transformados en Formato Largo (Panel Long Format):")
    st.dataframe(df_long.head(10))

    # Despliegue modular a través de pestañas funcionales
    tab1, tab2, tab3, tab4 = st.tabs(["Estadísticas Descriptivas", "Gráficos Interactivos", "Modelos Econométricos", "Exportar Resultados"])

    # =========================================================================
    # PESTAÑA 1: ESTADÍSTICAS DESCRIPTIVAS
    # =========================================================================
    with tab1:
        st.subheader("Estadísticas Descriptivas del Panel")
        desc_stats = df_long.describe().T[['mean', '50%', 'std', 'min', 'max']]
        desc_stats.columns = ['Media', 'Mediana', 'Desv. Estándar', 'Mínimo', 'Máximo']
        st.dataframe(desc_stats)

        st.subheader("Matriz de Correlaciones")
        df_numeric = df_long.select_dtypes(include=[np.number])
        corr_matrix = df_numeric.corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax_corr)
        st.pyplot(fig_corr)

    # =========================================================================
    # PESTAÑA 2: GRÁFICOS INTERACTIVOS Y DE DISPERSIÓN (DISEÑO ACADÉMICO)
    # =========================================================================
    with tab2:
        st.subheader("1. Evolución Temporal de Indicadores (Personalizable)")
        
        # --- SECCIÓN DE CONTROLES INTERACTIVOS ---
        col_ctrl1, col_ctrl2 = st.columns(2)
        
        with col_ctrl1:
            nivel_analisis = st.radio("Seleccione el nivel de análisis:", ["Promedio Regional", "País Específico"])
            if nivel_analisis == "País Específico":
                pais_seleccionado = st.selectbox("Seleccione un país:", df_long['País'].unique())
        
        with col_ctrl2:
            # Añadido CPI_SCORE dentro de las columnas disponibles para el gráfico
            columnas_grafico = [col for col in ['CPI_SCORE', 'RSF_SCORE', 'CORRUPCION', 'IED_PIB'] if col in df_long.columns]
            variables_seleccionadas = st.multiselect(
                "Seleccione las variables a graficar:", 
                options=columnas_grafico, 
                default=columnas_grafico # Por defecto muestra todas las disponibles incluyendo CPI
            )
            
        # --- GENERACIÓN DEL GRÁFICO DE LÍNEAS ---
        if not variables_seleccionadas:
            st.warning("Por favor, seleccione al menos una variable para visualizar.")
        else:
            # Filtrado de datos según nivel de análisis
            if nivel_analisis == "Promedio Regional":
                df_grafico = df_long.groupby('Año')[variables_seleccionadas].mean().reset_index()
                titulo = 'Evolución Promedio en la Región'
            else:
                df_grafico = df_long[df_long['País'] == pais_seleccionado]
                titulo = f'Trayectoria Temporal en {pais_seleccionado}'
                
            # Creación del gráfico
            fig_line = px.line(
                df_grafico, 
                x='Año', 
                y=variables_seleccionadas, 
                title=titulo, 
                markers=True,
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'] # 4 Colores sobrios institucionales
            )
            
            # --- DISEÑO ACADÉMICO / PROFESIONAL ---
            fig_line.update_layout(
                template='simple_white',      # Fondo blanco sin cuadrícula pesada
                title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'family': 'Arial', 'color': 'black'}},
                xaxis_title="Año",
                yaxis_title="Índice / Porcentaje",
                legend_title_text="Variables:",
                font=dict(family="Arial", size=13, color="black"),
                hovermode="x unified",        # Tooltip consolidado al pasar el ratón
                margin=dict(l=50, r=30, t=60, b=50),
                legend=dict(
                    orientation="h",          # Leyenda horizontal
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Engrosar líneas y marcadores para asegurar legibilidad en documentos impresos
            fig_line.update_traces(line=dict(width=2.5), marker=dict(size=8))
            
            # Mostrar el gráfico en Streamlit
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Botón de ayuda para el usuario
            st.caption("💡 *Tip: Puedes descargar este gráfico con calidad profesional haciendo clic en el icono de la cámara (Download plot as a png) que aparece al pasar el ratón por la esquina superior derecha del gráfico.*")

        st.markdown("---") # Separador visual

        st.subheader("2. Diagramas de Dispersión (Scatter Plots) con Línea de Tendencia")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_scatter1 = px.scatter(df_long, x='RSF_SCORE', y='CORRUPCION', hover_data=['País', 'Año'], trendline='ols', title="RSF vs Corrupción")
            st.plotly_chart(fig_scatter1, use_container_width=True)
        with col2:
            if 'IED_PIB' in df_long.columns:
                fig_scatter2 = px.scatter(df_long, x='CORRUPCION', y='IED_PIB', hover_data=['País', 'Año'], trendline='ols', title="Corrupción vs IED")
                st.plotly_chart(fig_scatter2, use_container_width=True)
        with col3:
            if 'IED_PIB' in df_long.columns:
                fig_scatter3 = px.scatter(df_long, x='RSF_SCORE', y='IED_PIB', hover_data=['País', 'Año'], trendline='ols', title="RSF vs IED")
                st.plotly_chart(fig_scatter3, use_container_width=True)

    # =========================================================================
    # PESTAÑA 3: ESTIMACIÓN DE MODELOS DINÁMICOS DE PANEL
    # =========================================================================
    with tab3:
        st.subheader("Estimación de Modelos de PanelOLS (Efectos Fijos de País y de Tiempo)")
        
        # Seteamos el MultiIndex (Índice de Grupo, Índice de Tiempo) requerido por linearmodels
        df_panel = df_long.set_index(['País', 'Año'])

        # Función enriquecida de interpretación automática orientada a la sustancia jurídico-económica
        def interpretar_resultado_sustantivo(modelo_num, variable, coef, p_value):
            significativo = p_value < 0.05
            
            texto = f"**Interpretación Institucional Automatizada ({variable}):**\n\n"
            if modelo_num == 1:
                if significativo:
                    if coef < 0:
                        texto += (f"El coeficiente asociado a **{variable}** es **negativo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Este hallazgo respalda empíricamente la hipótesis central de la investigación: un deterioro o disminución en las garantías a la libertad de prensa "
                                  "en el período anterior predice un incremento posterior en los niveles de corrupción sistémica. "
                                  "Desde una perspectiva jurídico-institucional, esto demuestra que restringir el ejercicio periodístico debilita los canales de fiscalización social, "
                                  "disminuye la probabilidad de detección pública y reduce sustancialmente los costos políticos y legales de las conductas corruptas.")
                    else:
                        texto += (f"El coeficiente asociado a **{variable}** es **positivo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Este resultado indica que incrementos en los índices de libertad de prensa se asocian temporalmente con un aumento en las métricas de corrupción. "
                                  "En la literatura especializada, esto se asocia al 'efecto destape': un entorno con mayor libertad informativa provee las condiciones para "
                                  "que los medios investiguen y expongan escándalos gubernamentales que antes permanecían ocultos bajo esquemas opacos, elevando temporalmente el registro de corrupción.")
                else:
                    texto += (f"El coeficiente de **{variable}** ({coef:.4f}) **no es estadísticamente significativo** (p={p_value:.3f} > 0.05). "
                              "Bajo la exigente especificación de efectos fijos bidimensionales, no se observa que las variaciones previas de la libertad de prensa "
                              "anticipen variaciones sistemáticas y lineales en los niveles de corrupción agregada para esta muestra de países.")
                    
            elif modelo_num == 2:
                if significativo:
                    if coef < 0:
                        texto += (f"El coeficiente asociado a **{variable}** es **negativo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Aporta fuerte evidencia sobre un ciclo destructivo de retroalimentación institucional: una mayor prevalencia de corrupción previa "
                                  "predice un menoscabo subsecuente sobre los puntajes de libertad de prensa (menor RSF_SCORE). Esto sugiere que las redes de corrupción institucionalizadas "
                                  "utilizan activamente el poder del Estado u otros mecanismos de presión para censurar, capturar o asfixiar financieramente a los medios digitales e impresos, "
                                  "buscando neutralizar las amenazas informativas a su impunidad.")
                    else:
                        texto += (f"El coeficiente asociado a **{variable}** es **positivo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Indica que mayores niveles de corrupción se asocian temporalmente con ganancias posteriores en libertad de prensa, sugiriendo dinámicas "
                                  "de resistencia civil donde la crisis institucional incentiva el surgimiento de periodismo independiente de investigación.")
                else:
                    texto += (f"El coeficiente de **{variable}** ({coef:.4f}) **no es estadísticamente significativo** (p={p_value:.3f} > 0.05). "
                              "Controlando estrictamente por diferencias estructurales internas y shocks macro-regionales comunes, la corrupción previa no ejerce un "
                              "efecto predictivo robusto sobre los niveles agregados de libertad de prensa.")

            elif modelo_num == 3:
                if significativo:
                    if coef < 0:
                        texto += (f"El coeficiente asociado a **{variable}** es **negativo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Este hallazgo valida la tesis económica del riesgo soberano: la corrupción previa actúa como un 'impuesto directo implícito' que ahuyenta los capitales. "
                                  "Al deteriorar la seguridad jurídica y distorsionar los contratos de mercado, un incremento de la opacidad institucional deprime de forma directa "
                                  "la atracción de Inversión Extranjera Directa (IED) en los periodos subsecuentes.")
                    else:
                        texto += (f"El coeficiente asociado a **{variable}** es **positivo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Este resultado se alinea teóricamente con la hipótesis de 'engrasar las ruedas' (*greasing the wheels*), la cual plantea que, "
                                  "en entornos con burocracias estatales extremadamente ineficientes, ciertos flujos internacionales de capital de corto plazo utilizan canales "
                                  "irregulares para agilizar transacciones operativas, comprometiendo la sostenibilidad de largo plazo.")
                else:
                    texto += (f"El coeficiente de **{variable}** ({coef:.4f}) **no es estadísticamente significativo** (p={p_value:.3f} > 0.05). "
                              "Controlando por factores temporales globales (como oscilaciones en tasas de interés internacionales o crisis sanitarias), la corrupción interna rezagada "
                              "no exhibe un impacto sistemático lineal sobre las fluctuaciones de la inversión extranjera.")

            elif modelo_num == 4:
                if significativo:
                    if coef > 0:
                        texto += (f"El coeficiente asociado a **{variable}** es **positivo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Este hallazgo es de alto valor estratégico para la política pública: un entorno transparente y seguro para el ejercicio de la prensa "
                                  "genera externalidades económicas directas y positivas. Al mitigar sistemáticamente las asimetrías de información en los mercados y servir como "
                                  "una señal creíble de robustez democrática, la libertad de prensa cataliza e impulsa la radicación de flujos de IED de manera autónoma.")
                    else:
                        texto += (f"El coeficiente asociado a **{variable}** es **negativo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Indica que aumentos en la libertad de prensa preceden contracciones en los flujos de IED, lo que refleja una postura transitoria de cautela por parte "
                                  "de las corporaciones globales frente a las turbulencias o debates políticos abiertos que se visibilizan a través de los medios masivos.")
                else:
                    texto += (f"El coeficiente de **{variable}** ({coef:.4f}) **no es estadísticamente significativo** (p={p_value:.3f} > 0.05). "
                              "El modelo bidimensional no halla un impacto directo de la libertad de prensa previa sobre la IED, lo cual sugiere que los efectos económicos de los "
                              "flujos de información operan de forma indirecta, estando potencialmente mediados por el control institucional de la corrupción.")
            return texto

        # Diccionarios de almacenamiento para la correcta compilación de reportes
        resultados_exportacion = {}
        interpretaciones_exportacion = {}

        # ---------------------------------------------------------------------
        # MODELO 1: Impacto de la Libertad de Prensa sobre la Corrupción
        # ---------------------------------------------------------------------
        st.markdown("### Modelo 1: Efecto de Libertad de Prensa sobre Corrupción")
        vars_mod1 = ['CORRUPCION', 'RSF_L1', 'CORRUPCION_L1']
        df_m1 = df_panel[vars_mod1].dropna()
        if not df_m1.empty:
            X1 = sm.add_constant(df_m1[['RSF_L1', 'CORRUPCION_L1']])
            Y1 = df_m1['CORRUPCION']
            
            # Estimación con efectos fijos individuales y temporales (entity_effects y time_effects)
            mod1 = PanelOLS(Y1, X1, entity_effects=True, time_effects=True)
            res1 = mod1.fit(cov_type='robust')
            st.text(res1.summary)
            
            interp1 = interpretar_resultado_sustantivo(1, "RSF_L1", res1.params['RSF_L1'], res1.pvalues['RSF_L1'])
            st.info("💡 **Interpretación Sustantiva:**\n\n" + interp1)
            resultados_exportacion['Modelo 1'] = res1.summary.as_text()
            interpretaciones_exportacion['Modelo 1'] = interp1
        else:
            st.warning("No hay suficientes datos válidos para estimar el Modelo 1.")

        # ---------------------------------------------------------------------
        # MODELO 2: Retroalimentación (Corrupción sobre Libertad de Prensa)
        # ---------------------------------------------------------------------
        st.markdown("### Modelo 2: Retroalimentación (Corrupción sobre Libertad de Prensa)")
        vars_mod2 = ['RSF_SCORE', 'CORRUPCION_L1', 'RSF_L1']
        df_m2 = df_panel[vars_mod2].dropna()
        if not df_m2.empty:
            X2 = sm.add_constant(df_m2[['CORRUPCION_L1', 'RSF_L1']])
            Y2 = df_m2['RSF_SCORE']
            
            mod2 = PanelOLS(Y2, X2, entity_effects=True, time_effects=True)
            res2 = mod2.fit(cov_type='robust')
            st.text(res2.summary)
            
            interp2 = interpretar_resultado_sustantivo(2, "CORRUPCION_L1", res2.params['CORRUPCION_L1'], res2.pvalues['CORRUPCION_L1'])
            st.info("💡 **Interpretación Sustantiva:**\n\n" + interp2)
            resultados_exportacion['Modelo 2'] = res2.summary.as_text()
            interpretaciones_exportacion['Modelo 2'] = interp2

        # ---------------------------------------------------------------------
        # MODELO 3: Efecto de la Corrupción sobre la Inversión Extranjera Directa
        # ---------------------------------------------------------------------
        st.markdown("### Modelo 3: Efecto de Corrupción sobre IED")
        if 'IED_PIB' in df_panel.columns and 'IED_L1' in df_panel.columns:
            vars_mod3 = ['IED_PIB', 'CORRUPCION_L1', 'IED_L1']
            
            # El dropna() aísla celdas vacías sin comprometer el resto del panel.
            df_m3 = df_panel[vars_mod3].dropna()
            if not df_m3.empty:
                X3 = sm.add_constant(df_m3[['CORRUPCION_L1', 'IED_L1']])
                Y3 = df_m3['IED_PIB']
                
                mod3 = PanelOLS(Y3, X3, entity_effects=True, time_effects=True)
                res3 = mod3.fit(cov_type='robust')
                st.text(res3.summary)
                
                interp3 = interpretar_resultado_sustantivo(3, "CORRUPCION_L1", res3.params['CORRUPCION_L1'], res3.pvalues['CORRUPCION_L1'])
                st.info("💡 **Interpretación Sustantiva:**\n\n" + interp3)
                resultados_exportacion['Modelo 3'] = res3.summary.as_text()
                interpretaciones_exportacion['Modelo 3'] = interp3
            else:
                st.warning("No hay suficientes observaciones de IED completas para procesar el Modelo 3.")

        # ---------------------------------------------------------------------
        # MODELO 4: Efecto Directo de la Libertad de Prensa sobre la IED
        # ---------------------------------------------------------------------
        st.markdown("### Modelo 4: Efecto de Libertad de Prensa sobre IED")
        if 'IED_PIB' in df_panel.columns and 'IED_L1' in df_panel.columns:
            vars_mod4 = ['IED_PIB', 'RSF_L1', 'IED_L1']
            
            df_m4 = df_panel[vars_mod4].dropna()
            if not df_m4.empty:
                X4 = sm.add_constant(df_m4[['RSF_L1', 'IED_L1']])
                Y4 = df_m4['IED_PIB']
                
                mod4 = PanelOLS(Y4, X4, entity_effects=True, time_effects=True)
                res4 = mod4.fit(cov_type='robust')
                st.text(res4.summary)
                
                interp4 = interpretar_resultado_sustantivo(4, "RSF_L1", res4.params['RSF_L1'], res4.pvalues['RSF_L1'])
                st.info("💡 **Interpretación Sustantiva:**\n\n" + interp4)
                resultados_exportacion['Modelo 4'] = res4.summary.as_text()
                interpretaciones_exportacion['Modelo 4'] = interp4
            else:
                st.warning("No existen suficientes registros de IED válidos para el Modelo 4.")

    # =========================================================================
    # PESTAÑA 4: COMPILACIÓN Y EXPORTACIÓN DE REPORTES INSTITUCIONALES
    # =========================================================================
    with tab4:
        st.subheader("Módulo de Descargas e Informes Académicos")
        
        # Canal de descarga para el dataframe estructurado en formato largo
        output_excel = io.BytesIO()
        df_long.to_excel(output_excel, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Descargar Base de Datos Transformada (Excel - Formato Largo)",
            data=output_excel.getvalue(),
            file_name="Panel_LATAM_LongFormat.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Compilación del reporte ejecutivo dinámico en Word
        def generar_word_reporte():
            doc = docx.Document()
            doc.add_heading('Informe de Investigación: Libertad de Prensa, Calidad Institucional y Desempeño Económico', 0)
            
            doc.add_heading('1. Análisis Descriptivo Agregado del Panel', level=1)
            doc.add_paragraph("A continuación se presentan las métricas de tendencia central y dispersión calculadas de forma global para la muestra:")
            doc.add_paragraph(desc_stats.to_string())
            
            doc.add_heading('2. Estimaciones con Modelos de Panel Dinámico (Doble Efecto Fijo)', level=1)
            doc.add_paragraph("Los coeficientes se computaron controlando simultáneamente por efectos fijos individuales (país) y choques temporales comunes (año), aplicando matrices de covarianza robustas:")
            
            for nombre_modelo, resumen_texto in resultados_exportacion.items():
                doc.add_heading(nombre_modelo, level=2)
                doc.add_paragraph("**Métricas Estadísticas del Modelo:**")
                doc.add_paragraph(resumen_texto)
                if nombre_modelo in interpretaciones_exportacion:
                    doc.add_paragraph("**Evaluación de la Lógica Sustantiva e Impacto Institucional:**")
                    doc.add_paragraph(interpretaciones_exportacion[nombre_modelo])
            
            word_io = io.BytesIO()
            doc.save(word_io)
            return word_io.getvalue()
            
        word_data = generar_word_reporte()
        st.download_button(
            label="📄 Descargar Informe Completo e Interpretación Sustantiva (Word)",
            data=word_data,
            file_name="Reporte_Final_Econometrico.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
