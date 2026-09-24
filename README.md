# Taller 1 - Machine Learning Aplicado (Equipo 3)

Repositorio del Taller 1 del curso de Machine Learning: dos notebooks que cubren el flujo completo de un proyecto de aprendizaje supervisado, desde el EDA hasta la evaluacion final en test, sobre dos problemas distintos.

## Integrantes

Jorge Andres Duran y Juan Miguel Castro

## Contenido

**Taller1_Regresion_Uber_1.ipynb**: Prediccion de tarifas de viajes de Uber en Nueva York (~200.000 registros). Incluye ingenieria de features (distancia de Haversine, codificacion ciclica de hora/mes, flag de aeropuerto), limpieza basada en dominio fisico (se conservan deliberadamente los outliers de negocio: viajes a aeropuerto y surge pricing), y comparacion de KNN, Ridge y Lasso con ajuste de hiperparametros via RandomizedSearchCV.

**Taller1_Clasificacion_Thyroid_1.ipynb**: Prediccion de recurrencia de cancer de tiroides. Incluye exclusion deliberada de la variable Response por fuga de informacion (es posterior al tratamiento), encoding ordinal con orden clinico explicito, y comparacion de KNN y Regresion Logistica (L1/L2), seleccionando el modelo final por recall en lugar de F1, dado el mayor costo clinico de los falsos negativos.

Ambos notebooks siguen la misma estructura: EDA, limpieza, split train/val/test, pipelines con ColumnTransformer, verificacion de que no hay fuga de datos, entrenamiento y diagnostico de sobreajuste/subajuste, evaluacion en test, tuning de hiperparametros, prediccion sobre un caso inventado y conclusiones.

## Por que los mismos algoritmos ganan en un caso y pierden en el otro

Un mismo trio de modelos (vecinos cercanos vs. modelos lineales regularizados) produce ganadores opuestos en los dos problemas, y la razon es la relacion entre el numero de observaciones y la dimensionalidad: con cerca de 135.000 observaciones de entrenamiento y una relacion no lineal, KNN gana en Uber; con apenas 254 observaciones y una relacion monotona, la Regresion Logistica regularizada gana en Thyroid. El sesgo de un modelo lineal deja de ser una debilidad y se vuelve una ventaja cuando la muestra es pequena.

## Como ejecutar

Ambos notebooks ya estan ejecutados, con todas las salidas y graficas embebidas. Para volver a correrlos desde cero se necesitan los datasets originales (uber.csv y Thyroid_Diff.csv) en la misma carpeta del notebook, o subidos a /content en Colab; la celda de carga busca en varias rutas comunes.
