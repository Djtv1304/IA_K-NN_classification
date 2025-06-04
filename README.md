# Informe de K-Vecinos más Cercanos (K-NN): Predicción de Rotación de Empleados

## 1. Introducción y Objetivo

El presente informe tiene como finalidad describir de manera detallada y estructurada el desarrollo, entrenamiento y evaluación de un **modelo de K-Vecinos más Cercanos (K-NN)** aplicado a datos de recursos humanos para predecir la **rotación de empleados** (`Attrition`). El objetivo concreto es:

* Utilizar variables demográficas y económicas (edad del empleado e ingreso mensual) para clasificar correctamente si un empleado permanecerá en la empresa (0) o la abandonará (1).
* Documentar el procedimiento completo de preprocesamiento, entrenamiento, evaluación y visualización del modelo, resaltando las métricas clave, especialmente la **precisión (accuracy)**.

## 2. Descripción del Conjunto de Datos

El dataset emplea dos variables predictoras y una variable objetivo, con los siguientes atributos:

| Variable           | Descripción                                                                      | Tipo                 |
| ------------------ | -------------------------------------------------------------------------------- | -------------------- |
| **Age**            | Edad del empleado (en años).                                                     | Numérica             |
| **MonthlyIncome**  | Ingreso mensual del empleado (en unidades monetarias).                          | Numérica             |
| **Attrition**      | Indicador binario: 1 si el empleado abandona la empresa, 0 si permanece.        | Categórica (Binaria) |

### Ejemplos de Datos

| Age | MonthlyIncome | Attrition |
| --- | ------------- | --------- |
| 41  | 5993          | 1         |
| 49  | 5130          | 0         |
| 37  | 2090          | 1         |
| 33  | 2909          | 0         |
| 27  | 3468          | 0         |

## 3. Preprocesamiento de Datos

1. **Carga del CSV**
   ```python
   dataset = pd.read_csv('Employee_Attrition_Modified.csv')
   ```
   El archivo contiene múltiples registros de empleados con las columnas `Age`, `MonthlyIncome` y `Attrition`.

2. **Separación de características (X) y variable objetivo (y)**
   ```python
   X = dataset.iloc[:, :-1].values  # [Age, MonthlyIncome]
   y = dataset.iloc[:, -1].values   # [Attrition]
   ```

3. **División en Conjunto de Entrenamiento y Prueba**
   * **Proporción**: 75% entrenamiento / 25% prueba.
   * **Aleatoriedad fija** (`random_state = 0`) para reproducibilidad.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=0
   )
   ```

4. **Escalado de Características**
   * Se utiliza **StandardScaler** para normalizar `Age` y `MonthlyIncome`.
   * Esto es **crítico en K-NN** ya que el algoritmo se basa en cálculo de distancias, y las diferencias de escala pueden sesgar los resultados.
   ```python
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test  = sc.transform(X_test)
   ```

## 4. Entrenamiento del Modelo K-NN

1. **Configuración del Clasificador**
   ```python
   classifier = KNeighborsClassifier(
       n_neighbors=9, 
       metric='minkowski', 
       p=2
   )
   classifier.fit(X_train, y_train)
   ```
   * **K = 9**: Número de vecinos considerados para la clasificación (valor impar para evitar empates).
   * **Métrica Minkowski con p=2**: Equivale a la distancia euclidiana.
   * **Principio**: Para cada punto de prueba, se identifican los 9 vecinos más cercanos y se asigna la clase mayoritaria.

2. **Predicción de Ejemplo Puntual**
   * Se proyecta un caso de prueba: **edad = 42 años**, **ingreso mensual = 6250**.
   * El modelo devuelve `0` o `1` indicando si se clasifica como permanencia o rotación.
   ```python
   resultado = classifier.predict(sc.transform([[42, 6250]]))
   print(f"Predicción para edad 42 e ingreso mensual de 6250: {resultado}")
   ```

## 5. Evaluación y Métricas

### 5.1 Predicciones sobre el Conjunto de Prueba

Se obtienen las predicciones del modelo (`y_pred`) y se comparan con las etiquetas reales (`y_test`). Los resultados muestran patrones interesantes:

```
[[0 0]  # Predicción correcta: No rotación
 [0 0]  # Predicción correcta: No rotación
 [0 0]  # Predicción correcta: No rotación
 ...
 [1 1]  # Predicción correcta: Rotación
 [0 1]  # Error: Predijo no rotación, pero sí hubo rotación (Falso Negativo)
 [1 0]  # Error: Predijo rotación, pero no la hubo (Falso Positivo)
 ...]
```

Cada fila muestra `[predicción, valor_real]`. Los errores de clasificación se evidencian cuando estos valores difieren.

### 5.2 Matriz de Confusión

Se construye la matriz de confusión a partir de `y_test` y `y_pred`:

|                              | Predicción = 0 (No Rotación) | Predicción = 1 (Rotación) |
| ---------------------------- | ---------------------------: | ------------------------: |
| **Real = 0 (No Rotación)**   |                      **199** |                    **10** |
| **Real = 1 (Rotación)**      |                       **39** |                     **3** |

* **Verdaderos Negativos (TN)** = 199
* **Falsos Positivos (FP)** = 10  
* **Falsos Negativos (FN)** = 39
* **Verdaderos Positivos (TP)** = 3

### 5.3 Métrica de Precisión (Accuracy)

La precisión global se calcula como:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{3 + 199}{3 + 199 + 10 + 39} = \frac{202}{251} \approx 0{,}8048
$$

* **Valor**: 0.8048 ⇒ **80,48%**
* En términos prácticos, **80 de cada 100** predicciones sobre el conjunto de prueba resultan correctas.

## 6. Visualización de los Resultados

### 6.1 Conjunto de Entrenamiento

![image](https://github.com/user-attachments/assets/161bbb35-b21a-45cd-b64b-83abfcadf62e)

**Descripción**:
* El plano se graficó usando las características originales (`Edad` ↔ `Age`, `Ingreso Mensual` ↔ `MonthlyIncome`).
* La región en **verde** corresponde a la clase `1` (rotación), y la región en **rojo** a la clase `0` (permanencia).
* Los puntos rojos representan observaciones de empleados que permanecen, y los verdes los que abandonan la empresa.
* Las **fronteras de decisión** muestran las regiones donde K-NN clasifica cada punto según sus 9 vecinos más cercanos.

**Observaciones**:
* Se aprecia una **clara separación geográfica**: la mayoría de puntos verdes (rotación) se concentran en la zona de **ingresos bajos** (≤ 10,000).
* Los empleados de **ingresos altos** (> 15,000) predominantemente permanecen en la empresa (región roja).
* La **frontera de decisión** es suave y coherente, sin fragmentaciones excesivas gracias al valor K=9.

### 6.2 Conjunto de Prueba

![image](https://github.com/user-attachments/assets/ef077839-a8be-4e85-9acc-67925361f615)

**Descripción**:
* Misma lógica gráfica que en el conjunto de entrenamiento.
* Se observan algunos puntos que caen en zonas opuestas a su color, los cuales corresponden a **errores de clasificación**.

**Observaciones**:
* La **consistencia visual** entre entrenamiento y prueba es alta, indicando que el modelo no presenta sobreajuste significativo.
* Los errores se concentran principalmente en la **zona de transición** (ingresos medios, ~5,000-10,000), donde la decisión es más incierta.
* La **predominancia del color rojo** refleja el desbalance natural del dataset: más empleados permanecen que los que se van.

## 7. Análisis Detallado del Rendimiento

### 7.1 Interpretación de la Precisión (80,48%)

* Un **accuracy** del 80,48% es **sólido** para problemas de recursos humanos, donde múltiples factores no capturados (satisfacción laboral, oportunidades externas, situación personal) influyen en la rotación.
* Sin embargo, la métrica global oculta un **desbalance crítico** en la detección de rotación real.

### 7.2 Análisis de Sensibilidad y Especificidad

1. **Tasa de Verdaderos Positivos (Sensibilidad)**
   $$
   \text{Sensibilidad} = \frac{TP}{TP + FN} = \frac{3}{3 + 39} = 0{,}0714 \quad(7{,}14\%)
   $$
   * **Resultado crítico**: De los 42 empleados que realmente abandonaron la empresa, el modelo solo detectó correctamente 3.
   * Una sensibilidad del 7,14% significa que **93 de cada 100 empleados que se irán quedan sin detectar**.

2. **Tasa de Verdaderos Negativos (Especificidad)**
   $$
   \text{Especificidad} = \frac{TN}{TN + FP} = \frac{199}{199 + 10} = 0{,}9522 \quad(95{,}22\%)
   $$
   * Excelente especificidad (95,22%), indicando que el modelo es muy eficaz identificando empleados que **permanecerán** en la empresa.

### 7.3 Implicaciones del Desbalance de Clases

* El dataset presenta un **fuerte desbalance**: aproximadamente 83% de empleados no rotan vs 17% que sí lo hacen.
* K-NN, al basarse en vecinos mayoritarios, tiende a **favorecer la clase dominante** (permanencia), explicando la baja sensibilidad.
* La **alta precisión global** es engañosa, pues se debe principalmente a la correcta clasificación de la clase mayoritaria.

### 7.4 Distribución de Errores por Tipo

* **Falsos Negativos (FN = 39)**: **Mayor riesgo empresarial**. No detectar empleados en riesgo de rotación impide tomar medidas preventivas (retención, mejoras salariales).
* **Falsos Positivos (FP = 10)**: Menor impacto, pero pueden generar costos innecesarios en programas de retención para empleados ya comprometidos.

## 8. Recomendaciones y Mejoras Futuras

### 8.1 Manejo del Desbalance de Clases

1. **Técnicas de Balanceeo**
   * **SMOTE** (Synthetic Minority Oversampling Technique): Generar ejemplos sintéticos de la clase minoritaria.
   * **Under-sampling**: Reducir muestras de la clase mayoritaria.
   * **Class weights**: Asignar mayor peso a la clase minoritaria durante el entrenamiento.

2. **Métricas Alternativas**
   * **F1-Score**: Media armónica entre precisión y recall, más robusta ante desbalances.
   * **AUC-ROC**: Evalúa la capacidad discriminativa independientemente del umbral.
   * **Precision-Recall Curve**: Especialmente útil con clases desbalanceadas.

### 8.2 Optimización de Hiperparámetros

1. **Selección del valor K óptimo**
   * Implementar **validación cruzada** con diferentes valores de K (1, 3, 5, 7, 9, 11, 15).
   * Evaluar usando **F1-score** en lugar de accuracy para casos desbalanceados.

2. **Exploración de métricas de distancia**
   * Probar distancia **Manhattan** (p=1) o **Chebyshev** para datos con outliers.
   * **Distancias ponderadas** que den mayor importancia a características más relevantes.

### 8.3 Incorporación de Variables Adicionales

* **Variables demográficas**: Género, estado civil, nivel educativo.
* **Variables laborales**: Años en la empresa, nivel jerárquico, departamento, evaluaciones de desempeño.
* **Variables de satisfacción**: Encuestas de clima laboral, balance vida-trabajo.
* **Variables económicas**: Comparación salarial con mercado, bonificaciones, beneficios.

### 8.4 Modelos Alternativos

1. **Random Forest**: Maneja mejor el desbalance y captura interacciones no lineales.
2. **Gradient Boosting** (XGBoost, LightGBM): Excelente rendimiento en problemas de clasificación desbalanceada.
3. **SVM con kernel RBF**: Puede capturar patrones no lineales complejos.
4. **Redes Neuronales**: Con técnicas de regularización y balanceeo de clases.

### 8.5 Validación Robusta

* **Stratified K-Fold Cross-Validation**: Mantiene la proporción de clases en cada fold.
* **Time Series Split**: Si los datos tienen componente temporal, evitar data leakage.
* **Holdout final**: Conjunto de validación completamente independiente para evaluación final.

## 9. Conclusiones

* El modelo de **K-Vecinos más Cercanos** logró una **precisión (accuracy) del 80,48%**, con **sensibilidad crítica del 7,14%** y **especificidad excelente del 95,22%** en el conjunto de prueba.

* La **matriz de confusión** revela un problema severo de **falsos negativos** (39 casos), indicando que el modelo es **inadecuado para detectar empleados en riesgo de rotación**, que es precisamente el objetivo principal del negocio.

* Las **visualizaciones** muestran una separación coherente basada en ingresos, con empleados de bajos ingresos más propensos a la rotación, pero el **desbalance de clases** limita severamente la utilidad práctica del modelo.

* Para un entorno empresarial de recursos humanos, es **crítico maximizar la sensibilidad** para detectar empleados en riesgo, incluso a costa de reducir la precisión global.

* Se recomienda **implementar técnicas de balanceeo de clases, optimizar hiperparámetros usando métricas apropiadas para datos desbalanceados, e incorporar variables adicionales** que capturen mejor los factores que influyen en la decisión de rotación de empleados.

**Autor:** Diego Toscano<br>
**Contacto:** [diego.toscano@udla.edu.ec](mailto:diego.toscano@udla.edu.ec)
