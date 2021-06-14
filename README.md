# Framework para generación y visualización de ensambles de clasificadores

Esta herramienta permitirá la generación de ensambles de clasificadores de forma interactiva
Se usó umap para la reducción de dimensionalidad. Esto se hace para las visualizaciones, no para el entrenamiento

## Primero se visualiza con matplotlib

![Matplotlib](/matplotlib.jpg)

## Se creó una visualización interactiva con Bokeh de python
![Bokeh](/bokeh.jpg)

Los modelos se encuentran en main.py, son los siguientes:
- Decision tree
- Random forest
- Logistic Regression

Se probó con un conjunto de datos básico disponible en https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv
