# Framework para generación y visualización de ensambles de clasificadores

Esta herramienta permitirá la generación de ensambles de clasificadores de forma interactiva
Se usó umap para la reducción de dimensionalidad. Esto se hace para las visualizaciones, no para el entrenamiento

Esta herramienta permite crear ensambles, modificarlos, asignar pesos, guardarlos, cargarlos, ver las métricas
de cada modelo y ensamble, compararlos y ver las gráficas de los modelos.

Los modelos son los siguientes:
- Decision tree
- Random forest
- Logistic Regression

Se usa el conjunto de datos de digits, ya que es usado tambien por el trabajo que deseamos mejorar https://www.sciencedirect.com/science/article/abs/pii/S0097849319301402?via%3Dihub

El framework esta completado al 80%, sólo falta realizar lo siguiente:
1. Agregar mas modelos que puedan ser incluidos en los ensambles
2. Agregar metricas mas complejas para dar mas informacion
3. Graficar los limites de decision de los modelos

![alt text](https://github.com/agublazer/pfc2/main1.jpg?raw=true)
![alt text](https://github.com/agublazer/pfc2/blob/master/main2.jpg?raw=true)
