# =========================================================================
    # PESTAÑA 2: GRÁFICOS INTERACTIVOS Y DE DISPERSIÓN
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
            columnas_grafico = [col for col in ['RSF_SCORE', 'CORRUPCION', 'IED_PIB'] if col in df_long.columns]
            variables_seleccionadas = st.multiselect(
                "Seleccione las variables a graficar:", 
                options=columnas_grafico, 
                default=columnas_grafico # Por defecto muestra todas
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
                color_discrete_sequence=['#1f77b4', '#d62728', '#2ca02c'] # Colores sobrios
            )
            
            # --- DISEÑO ACADÉMICO / PROFESIONAL ---
            fig_line.update_layout(
                template='simple_white',      # Fondo blanco sin cuadrícula pesada
                title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'family': 'Arial', 'color': 'black'}},
                xaxis_title="Año",
                yaxis_title="Índice / Porcentaje",
                legend_title_text="Variables:",
                font=dict(family="Arial", size=13, color="black"),
                hovermode="x unified",        # Tooltip consolidado al pasar el ratón (muestra todos los valores del año)
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
