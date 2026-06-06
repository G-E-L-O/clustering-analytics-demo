# Análisis de Clustering Interactivo

Este repositorio contiene una aplicación interactiva desarrollada en **Python (Streamlit)** para analizar y visualizar el comportamiento matemático de los algoritmos de clustering más populares de la industria: **K-Means**, **DBSCAN** y **Clustering Jerárquico**.

El objetivo de la herramienta es demostrar de forma educativa y técnica cómo convergen estos modelos ante distintos escenarios del mundo real, y cómo aplicar métricas para validarlos.

## 🚀 Casos de Estudio Incluidos
1. **Segmentación de Clientes (Blobs):** Simulación de datos esféricos, ideal para evaluar el agrupamiento por inercia y centroides (Ej. Ingreso Anual vs Puntuación de Gasto).
2. **Análisis de Rutas Geográficas (Moons):** Simulación de datos basados en densidad y formas irregulares, ideal para demostrar la debilidad de K-Means frente a la superioridad de la conectividad de DBSCAN.

## ⚙️ Características Técnicas
* **K-Means Iterativo (Paso a Paso):** Implementación manual utilizando `scipy.spatial.distance.cdist` para calcular matrices de distancias Euclidianas y recalcular los centros de masa iteración por iteración.
* **K-Means Final:** Comparativa utilizando la implementación altamente optimizada de `scikit-learn` (K-Means++).
* **K-Means (Método del Codo):** Generación matemática de la curva de Inercia (WCSS) iterando modelos desde K=1 hasta K=10 para justificar el número óptimo de clusters.
* **DBSCAN (Density-Based):** Análisis de agrupamiento basado en vecindad (`eps`) y puntos mínimos (`minPts`), diferenciando visualmente Puntos Núcleo, Puntos Frontera y Outliers (Ruido).
* **Clustering Jerárquico (Dendrogramas):** Visualización del árbol de distancias de fusión (Linkage) permitiendo alternar entre métodos de enlace (`ward`, `complete`, `average`, `single`). Incorpora submuestreo inteligente automático para legibilidad en N > 100.
* **Evaluación Matemática:** Cálculo en tiempo real del **Coeficiente de Silhouette** para validar la cohesión y separación de los clusters formados.

## 💻 Instalación y Uso

1. Clona este repositorio:
```bash
git clone https://github.com/G-E-L-O/clustering-analytics-demo.git
cd clustering-analytics-demo
```

2. Instala las dependencias requeridas:
```bash
pip install -r requirements.txt
```

3. Ejecuta la aplicación interactiva:
```bash
streamlit run demo_clustering.py
```
