# Trabajo Final de Máster. Sistema de predicción de ventas semanales por tienda y departamento: Enfoque basado en CRISP-DM para la toma de decisiones

Este repositorio contiene el desarrollo del Trabajo Final de Máster (TFM), cuyo objetivo principal es la construcción de un modelo predictivo capaz de estimar la demanda 
semanal de ventas en un entorno minorista. El proyecto se enmarca dentro del máster en Big Data y Data Science y está orientado a la toma de decisiones estratégicas en la gestión de inventarios y recursos usando la metodologia CRISP-DM.

## Objetivo del Proyecto

El propósito de este trabajo es predecir el volumen de ventas semanales por tienda y por departamento, integrando distintas fuentes de información tales como:
- Históricos de ventas
- Eventos promocionales
- Variables económicas
- Datos temporales (semana del año, festivos, etc.)

La solución propuesta tiene como meta mejorar la planificación de inventarios, reducir pérdidas por exceso o falta de stock y ofrecer una herramienta de soporte a la toma de decisiones comerciales.

## Enfoque Metodológico: CRISP-DM
Para la estructura y ejecución del proyecto se ha seguido el marco metodológico **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, reconocido como estándar en procesos de minería de datos. Las fases desarrolladas fueron:

1. **Comprensión del Negocio:**  
   Se analizó el contexto operativo del sector retail, identificando como problema principal la previsión de demanda en diferentes niveles de granularidad (tienda y departamento).

2. **Comprensión de los Datos:**  
   Se realizó un análisis exploratorio detallado de las bases de datos provistas, evaluando la calidad de los datos, la existencia de valores atípicos y nulos, así como la distribución temporal y estacional de las variables.

3. **Preparación de los Datos:**  
   Incluyó la limpieza, transformación y normalización de variables, creación de variables derivadas (por ejemplo, fuerza promocional), y codificación de categorías necesarias para el modelado.

4. **Modelado:**  
   Se emplearon diferentes algoritmos de aprendizaje automático y series temporales:
   - Modelos clásicos: ARIMA, Prophet
   - Modelos supervisados: Random Forest, XGBoost, ExtraTrees
   Los modelos fueron evaluados mediante métricas como MAE, RMSE y R².

5. **Evaluación:**  
   Se seleccionó el modelo más robusto según su rendimiento general y específico por tienda y departamento. También se realizaron pruebas comparativas en semanas con eventos especiales y sin eventos.

6. **Despliegue:**  
   Se desarrolló una interfaz de usuario utilizando **Streamlit**, permitiendo realizar predicciones personalizadas por tienda o por departamento, mostrando resultados junto con un nivel de confiabilidad estimado.


## Estructura del Repositorio
├── TFM.ipynb # Notebook principal con el análisis completo

├── app_ventas_streamlit.py # Aplicación web para predicción de ventas

├── README.md # Descripción del proyecto

## Autor
Este proyecto ha sido realizado por Eduardo Carrasco Taboada, como parte de los requisitos académicos para la obtención del título de Máster en Big Data y Data Science.
