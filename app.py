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
# CONFIGURACIÓN DE LA INTERFAZ DE USUARIO Y BARRA LATERAL
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Análisis Econométrico de Panel", layout="wide")
st.title("📊 Análisis Econométrico: Libertad de Prensa, Corrupción e IED")
st.markdown("""
Esta aplicación permite transformar una base de datos, explorar descriptivamente las interacciones, 
identificar casos atípicos y estimar modelos de panel bidimensionales. Utiliza el menú lateral para ajustar los parámetros globales.
""")

st.sidebar.header("1. Carga de Archivos")
uploaded_data = st.sidebar.file_uploader("Sube tu base de datos (Excel o CSV)", type=["xlsx", "csv"])
uploaded_doc = st.sidebar.file_uploader("Sube el documento Word metodológico (Opcional)", type=["docx"])

if uploaded_doc is not None:
    st.sidebar.success("Documento Word cargado correctamente.")
    with st.sidebar.expander("📄 Ver Documento Metodológico"):
        doc = docx.Document(uploaded_doc)
        for para in doc.paragraphs:
            st.write(para.text)

st.sidebar.markdown("---")
st.sidebar.header("2. Configuración Global")
# TOGGLE MAESTRO: Controla si toda la app usa (t-1) o (t) para dispersión y modelos
aplicar_rezago_global = st.sidebar.checkbox(
    "🔄 Aplicar Rezago Temporal (t-1)", 
    value=True, 
    help="Si está activo, evalúa cómo el año anterior impacta al actual (Modelos Dinámicos). Si se desactiva, evalúa relaciones en el mismo año (Modelos Contemporáneos Estáticos)."
)

if aplicar_rezago_global:
    st.sidebar.info("Modo Dinámico Activado: Las variables explicativas tendrán un rezago de 1 año (t-1).")
else:
    st.sidebar.warning("Modo Estático Activado: Todas las relaciones se evalúan en el mismo año (t).")

# -----------------------------------------------------------------------------
# MOTOR DE PROCESAMIENTO ECONOMÉTRICO
# -----------------------------------------------------------------------------
if uploaded_data is not None:
    try:
        if uploaded_data.name.endswith('.csv'):
            df_wide = pd.read_csv(uploaded_data)
        else:
            df_wide = pd.read_excel(uploaded_data)
    except Exception as e:
        st.error(f"Error crítico al leer el archivo de entrada: {e}")
        st.stop()

    # Definición de una lista explícita de stubnames para máxima robustez
    stubnames_lista = ['CPI_SCORE', 'CPI_RANK', 'CPI_RANKING', 'CORRUPCION', 'RSF_SCORE', 'RSF_RANK', 'RSF_RANKING', 'IED_PIB']
    
    # Reestructuración limpia de formato Ancho (Wide) a Largo (Long)
    df_long = pd.wide_to_long(
        df_wide, 
        stubnames=stubnames_lista,
        i='País',
        j='Año',
        sep='_',
        suffix=r'\d+'
    ).reset_index()               

    # Ordenamiento jerárquico indispensable para el correcto cálculo de rezagos temporales
    df_long = df_long.sort_values(by=['País', 'Año'])

    # -------------------------------------------------------------------------
    # GENERACIÓN DE VARIABLES REZAGADAS (LAGGED VARIABLES)
    # -------------------------------------------------------------------------
    df_long['RSF_L1'] = df_long.groupby('País')['RSF_SCORE'].shift(1)
    df_long['CORRUPCION_L1'] = df_long.groupby('País')['CORRUPCION'].shift(1)
    
    if 'IED_PIB' in df_long.columns:
        df_long['IED_L1'] = df_long.groupby('País')['IED_PIB'].shift(1)
    else:
        df_long['IED_L1'] = np.nan

    # Asignación dinámica de variables explicativas basadas en el Toggle Maestro
    var_rsf_explicativa = 'RSF_L1' if aplicar_rezago_global else 'RSF_SCORE'
    var_corr_explicativa = 'CORRUPCION_L1' if aplicar_rezago_global else 'CORRUPCION'
    sufijo_tiempo = "(t-1)" if aplicar_rezago_global else "(t)"

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
    # PESTAÑA 2: GRÁFICOS INTERACTIVOS, CASOS ATÍPICOS Y DISPERSIÓN
    # =========================================================================
    with tab2:
        st.subheader("1. Evolución Temporal de Indicadores (Personalizable)")
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            nivel_analisis = st.radio("Seleccione el nivel de análisis:", ["Promedio Regional", "País Específico"])
            if nivel_analisis == "País Específico":
                pais_seleccionado = st.selectbox("Seleccione un país:", df_long['País'].unique())
        
        with col_ctrl2:
            columnas_grafico = [col for col in ['CPI_SCORE', 'RSF_SCORE', 'CORRUPCION', 'IED_PIB'] if col in df_long.columns]
            variables_seleccionadas = st.multiselect(
                "Seleccione las variables a graficar:", 
                options=columnas_grafico, 
                default=columnas_grafico
            )
            
        if not variables_seleccionadas:
            st.warning("Por favor, seleccione al menos una variable para visualizar.")
        else:
            if nivel_analisis == "Promedio Regional":
                df_grafico = df_long.groupby('Año')[variables_seleccionadas].mean().reset_index()
                titulo = 'Evolución Promedio en la Región'
            else:
                df_grafico = df_long[df_long['País'] == pais_seleccionado]
                titulo = f'Trayectoria Temporal en {pais_seleccionado}'
                
            fig_line = px.line(
                df_grafico, x='Año', y=variables_seleccionadas, title=titulo, markers=True,
                color_discrete_sequence=['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'] 
            )
            
            fig_line.update_layout(
                template='simple_white',
                title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'family': 'Arial', 'color': 'black'}},
                xaxis_title="Año", yaxis_title="Índice / Porcentaje", legend_title_text="Variables:",
                font=dict(family="Arial", size=13, color="black"), hovermode="x unified",
                margin=dict(l=50, r=30, t=60, b=50), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_line.update_traces(line=dict(width=2.5), marker=dict(size=8))
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")
        
        # --- NUEVO MÓDULO: CASOS ATÍPICOS Y VARIACIONES ---
        st.subheader("2. Casos Atípicos y Análisis de Variación (Ganadores y Perdedores)")
        st.write("Visualiza rápidamente qué países registraron los cambios más extremos (positivos o negativos) comparando su primer año de registro con su último año.")
        
        var_atipica = st.selectbox("Seleccione la variable para analizar casos atípicos:", columnas_grafico)
        
        # Cálculo de las variaciones extremas
        df_agrupado = df_long.dropna(subset=[var_atipica]).sort_values(by=['País', 'Año'])
        primeros_registros = df_agrupado.groupby('País').first().reset_index()
        ultimos_registros = df_agrupado.groupby('País').last().reset_index()
        
        df_variaciones = pd.DataFrame({
            'País': primeros_registros['País'],
            'Año Inicial': primeros_registros['Año'],
            'Valor Inicial': primeros_registros[var_atipica],
            'Año Final': ultimos_registros['Año'],
            'Valor Final': ultimos_registros[var_atipica],
            'Variación Neta': ultimos_registros[var_atipica] - primeros_registros[var_atipica]
        }).sort_values(by='Variación Neta', ascending=False)
        
        # Gráfico de barras divergente
        fig_var = px.bar(
            df_variaciones, 
            x='País', 
            y='Variación Neta', 
            title=f"Variación Total de {var_atipica} en el Período Estudiado",
            color='Variación Neta',
            color_continuous_scale=px.colors.diverging.RdYlGn, # Escala de Rojo (Negativo) a Verde (Positivo)
            text_auto='.2f'
        )
        fig_var.update_layout(
            template='simple_white', 
            font=dict(family="Arial", color="black"),
            xaxis_title="", 
            yaxis_title="Cambio Neto (Puntos/Porcentaje)",
            coloraxis_showscale=False # Ocultar la barra de color lateral para un look más académico
        )
        st.plotly_chart(fig_var, use_container_width=True)
        
        # Tablas de resumen para los Top Extremos
        col_ganadores, col_perdedores = st.columns(2)
        with col_ganadores:
            st.success(f"🏆 Top 3: Mayores Incrementos en {var_atipica}")
            # Formateo limpio para mostrar en pantalla
            st.dataframe(df_variaciones.head(3)[['País', 'Valor Inicial', 'Valor Final', 'Variación Neta']].style.format({"Valor Inicial": "{:.2f}", "Valor Final": "{:.2f}", "Variación Neta": "{:+.2f}"}))
        with col_perdedores:
            st.error(f"🚨 Top 3: Caídas Más Extremas en {var_atipica}")
            st.dataframe(df_variaciones.tail(3)[['País', 'Valor Inicial', 'Valor Final', 'Variación Neta']].style.format({"Valor Inicial": "{:.2f}", "Valor Final": "{:.2f}", "Variación Neta": "{:+.2f}"}))

        st.markdown("---")
        st.subheader(f"3. Diagramas de Dispersión (Dependen del Toggle Maestro: {sufijo_tiempo})")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fig_scatter1 = px.scatter(df_long, x=var_rsf_explicativa, y='CORRUPCION', hover_data=['País', 'Año'], trendline='ols', 
                                      title=f"RSF {sufijo_tiempo} vs Corrupción (t)")
            fig_scatter1.update_layout(template='simple_white', font=dict(family="Arial", color="black"))
            st.plotly_chart(fig_scatter1, use_container_width=True)
            
        with col2:
            if 'IED_PIB' in df_long.columns:
                fig_scatter2 = px.scatter(df_long, x=var_corr_explicativa, y='IED_PIB', hover_data=['País', 'Año'], trendline='ols', 
                                          title=f"Corrupción {sufijo_tiempo} vs IED (t)")
                fig_scatter2.update_layout(template='simple_white', font=dict(family="Arial", color="black"))
                st.plotly_chart(fig_scatter2, use_container_width=True)
                
        with col3:
            if 'IED_PIB' in df_long.columns:
                fig_scatter3 = px.scatter(df_long, x=var_rsf_explicativa, y='IED_PIB', hover_data=['País', 'Año'], trendline='ols', 
                                          title=f"RSF {sufijo_tiempo} vs IED (t)")
                fig_scatter3.update_layout(template='simple_white', font=dict(family="Arial", color="black"))
                st.plotly_chart(fig_scatter3, use_container_width=True)

    # =========================================================================
    # PESTAÑA 3: ESTIMACIÓN DE MODELOS (ESTÁTICOS O DINÁMICOS SEGÚN TOGGLE)
    # =========================================================================
    with tab3:
        if aplicar_rezago_global:
            st.subheader("Estimación de Modelos de Panel DINÁMICOS (Efectos Fijos Bidimensionales con Lags)")
        else:
            st.subheader("Estimación de Modelos de Panel ESTÁTICOS CONTEMPORÁNEOS (Efectos Fijos Bidimensionales)")
            
        df_panel = df_long.set_index(['País', 'Año'])

        def interpretar_resultado_sustantivo(modelo_num, variable, coef, p_value, usar_rezago):
            significativo = p_value < 0.05
            
            t_anterior = "en el período anterior predice un incremento posterior" if usar_rezago else "se asocia simultáneamente con un incremento"
            t_previa = "previa" if usar_rezago else "contemporánea"
            t_subsecuente = "subsecuente" if usar_rezago else "inmediato"
            t_preceden = "preceden contracciones" if usar_rezago else "están correlacionados con contracciones"
            t_ahuyenta = "previa actúa como un 'impuesto directo implícito' que ahuyenta" if usar_rezago else "actúa como un 'impuesto directo implícito' que frena inmediatamente"
            
            texto = f"**Interpretación Institucional Automatizada ({variable}):**\n\n"
            if modelo_num == 1:
                if significativo:
                    if coef < 0:
                        texto += (f"El coeficiente asociado a **{variable}** es **negativo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  f"Este hallazgo indica que un deterioro en las garantías a la libertad de prensa {t_anterior} en los niveles de corrupción sistémica. "
                                  "Desde una perspectiva jurídico-institucional, esto demuestra que restringir el ejercicio periodístico debilita los canales de fiscalización social.")
                    else:
                        texto += (f"El coeficiente asociado a **{variable}** es **positivo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Este resultado indica que incrementos en los índices de libertad de prensa se asocian con un aumento en las métricas de corrupción. "
                                  "En la literatura, esto se asocia al 'efecto destape': un entorno de libertad provee condiciones para exponer escándalos previamente ocultos.")
                else:
                    texto += f"El coeficiente de **{variable}** ({coef:.4f}) **no es estadísticamente significativo** (p={p_value:.3f} > 0.05). No se observa una relación sistemática bajo esta especificación."
                    
            elif modelo_num == 2:
                if significativo:
                    if coef < 0:
                        texto += (f"El coeficiente es **negativo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  f"Evidencia un ciclo destructivo: una mayor prevalencia de corrupción {t_previa} genera un menoscabo {t_subsecuente} sobre la libertad de prensa. "
                                  "Sugiere que redes institucionalizadas utilizan el poder del Estado para asfixiar a los medios y proteger su impunidad.")
                    else:
                        texto += (f"El coeficiente es **positivo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Mayores niveles de corrupción se asocian con ganancias en libertad de prensa, sugiriendo dinámicas de resistencia civil.")
                else:
                    texto += f"El coeficiente de **{variable}** ({coef:.4f}) **no es estadísticamente significativo** (p={p_value:.3f} > 0.05)."

            elif modelo_num == 3:
                if significativo:
                    if coef < 0:
                        texto += (f"El coeficiente es **negativo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  f"Valida la tesis del riesgo soberano: la corrupción {t_ahuyenta} los capitales al deteriorar la seguridad jurídica de los contratos de mercado.")
                    else:
                        texto += (f"El coeficiente es **positivo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Se alinea con la hipótesis de 'engrasar las ruedas', donde el capital utiliza canales irregulares para agilizar burocracias ineficientes.")
                else:
                    texto += f"El coeficiente de **{variable}** ({coef:.4f}) **no es estadísticamente significativo** (p={p_value:.3f} > 0.05). La opacidad institucional no exhibe un impacto lineal directo sobre la IED."

            elif modelo_num == 4:
                if significativo:
                    if coef > 0:
                        texto += (f"El coeficiente es **positivo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  "Un entorno transparente y seguro para la prensa genera externalidades económicas positivas. Al mitigar asimetrías de información, cataliza la IED.")
                    else:
                        texto += (f"El coeficiente es **negativo** ({coef:.4f}) y **estadísticamente significativo** (p={p_value:.3f} < 0.05). "
                                  f"Aumentos en la libertad de prensa {t_preceden} en flujos de IED, reflejando una postura transitoria de cautela corporativa frente a turbulencias políticas visibles.")
                else:
                    texto += f"El coeficiente de **{variable}** ({coef:.4f}) **no es estadísticamente significativo** (p={p_value:.3f} > 0.05)."
            return texto

        resultados_exportacion, interpretaciones_exportacion = {}, {}

        # ---------------------------------------------------------------------
        # ESTRUCTURACIÓN DE MODELOS
        # ---------------------------------------------------------------------
        def correr_modelo(num_mod, titulo, var_dep, vars_indep, var_foco):
            st.markdown(f"### {titulo}")
            variables_necesarias = [var_dep] + vars_indep
            df_m = df_panel[variables_necesarias].dropna()
            
            if not df_m.empty:
                X = sm.add_constant(df_m[vars_indep])
                Y = df_m[var_dep]
                mod = PanelOLS(Y, X, entity_effects=True, time_effects=True)
                res = mod.fit(cov_type='robust')
                st.text(res.summary)
                
                interp = interpretar_resultado_sustantivo(num_mod, var_foco, res.params[var_foco], res.pvalues[var_foco], aplicar_rezago_global)
                st.info("💡 **Interpretación Sustantiva:**\n\n" + interp)
                
                resultados_exportacion[titulo] = res.summary.as_text()
                interpretaciones_exportacion[titulo] = interp
            else:
                st.warning(f"No hay suficientes datos válidos para estimar el Modelo {num_mod}.")

        # MODELO 1
        vars_indep_m1 = ['RSF_L1', 'CORRUPCION_L1'] if aplicar_rezago_global else ['RSF_SCORE']
        correr_modelo(1, "Modelo 1: Efecto de Libertad de Prensa sobre Corrupción", 'CORRUPCION', vars_indep_m1, var_rsf_explicativa)

        # MODELO 2
        vars_indep_m2 = ['CORRUPCION_L1', 'RSF_L1'] if aplicar_rezago_global else ['CORRUPCION']
        correr_modelo(2, "Modelo 2: Efecto de Corrupción sobre Libertad de Prensa", 'RSF_SCORE', vars_indep_m2, var_corr_explicativa)

        # MODELO 3
        if 'IED_PIB' in df_panel.columns:
            vars_indep_m3 = ['CORRUPCION_L1', 'IED_L1'] if aplicar_rezago_global else ['CORRUPCION']
            correr_modelo(3, "Modelo 3: Efecto de Corrupción sobre IED", 'IED_PIB', vars_indep_m3, var_corr_explicativa)

        # MODELO 4
        if 'IED_PIB' in df_panel.columns:
            vars_indep_m4 = ['RSF_L1', 'IED_L1'] if aplicar_rezago_global else ['RSF_SCORE']
            correr_modelo(4, "Modelo 4: Efecto de Libertad de Prensa sobre IED", 'IED_PIB', vars_indep_m4, var_rsf_explicativa)

    # =========================================================================
    # PESTAÑA 4: COMPILACIÓN Y EXPORTACIÓN DE REPORTES INSTITUCIONALES
    # =========================================================================
    with tab4:
        st.subheader("Módulo de Descargas e Informes Académicos")
        
        output_excel = io.BytesIO()
        df_long.to_excel(output_excel, index=False, engine='openpyxl')
        st.download_button(
            label="📥 Descargar Base de Datos Transformada (Excel - Formato Largo)",
            data=output_excel.getvalue(),
            file_name="Panel_LATAM_LongFormat.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        def generar_word_reporte():
            doc = docx.Document()
            doc.add_heading('Informe de Investigación: Libertad de Prensa, Calidad Institucional y Desempeño Económico', 0)
            
            tipo_mod = "Dinámicos con Rezagos (t-1)" if aplicar_rezago_global else "Estáticos Contemporáneos (t)"
            doc.add_paragraph(f"Nota Metodológica: Para la estructuración de este reporte se utilizó el modelo de {tipo_mod}.")
            
            doc.add_heading('1. Análisis Descriptivo Agregado del Panel', level=1)
            doc.add_paragraph(desc_stats.to_string())
            
            doc.add_heading('2. Estimaciones con Modelos de Panel', level=1)
            
            for nombre_modelo, resumen_texto in resultados_exportacion.items():
                doc.add_heading(nombre_modelo, level=2)
                doc.add_paragraph(resumen_texto)
                if nombre_modelo in interpretaciones_exportacion:
                    doc.add_paragraph(interpretaciones_exportacion[nombre_modelo])
            
            word_io = io.BytesIO()
            doc.save(word_io)
            return word_io.getvalue()
            
        word_data = generar_word_reporte()
        st.download_button(
            label="📄 Descargar Informe Completo (Word)",
            data=word_data,
            file_name="Reporte_Final_Econometrico.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
