# Sistema de transacciones de clientes

Este proyecto desarrolla un sistema de recomendación de productos basado en datos de transacciones de clientes. A través de técnicas de procesamiento de datos, preprocesamiento de texto y modelado de similitud, se generan recomendaciones personalizadas basadas en el historial de transacciones de cada cliente.

## Requisitos

Asegúrate de tener las siguientes bibliotecas instaladas:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `nltk`
- `scikit-learn`

Además, algunas dependencias de NLTK deben descargarse si no están disponibles:

```python
nltk.download('stopwords')
nltk.download('wordnet')

Estructura del Código
1. Importaciones
Las bibliotecas necesarias para el análisis de datos, visualización, procesamiento de texto, y modelado de recomendaciones son importadas.

2. Carga y Limpieza de Datos
Los datos son cargados desde un archivo CSV.
Se eliminan duplicados para evitar datos redundantes.
Se rellenan valores nulos en la columna Gender con "Unknown" y en Transaction Amount con la mediana.
Los valores nulos en otras columnas se rellenan con un valor genérico "N/A".
3. Análisis Exploratorio de Datos (EDA)
Se realizan visualizaciones para explorar el conjunto de datos:

Distribución de transacciones por categoría de producto.
Distribución del monto de transacciones.
Relación entre el monto de transacciones y el género.
4. Preprocesamiento de Datos
Codificación de datos: Gender es codificado como variable numérica y Category como variables dummy.
Preprocesamiento de texto: se lematiza y limpia la columna Merchant Name para eliminar palabras vacías y caracteres especiales.
5. Transformación de Texto con TF-IDF
El texto de Merchant Name se vectoriza mediante TF-IDF para representar las palabras clave de cada transacción, con un máximo de 100 características.

6. Creación de la Matriz de Similitud
Se crea una matriz de similitud de coseno basada en las características TF-IDF, que se utiliza para calcular la similitud entre las transacciones.

7. Función de Recomendación
La función get_recommendations recibe un customer_id y devuelve una lista de recomendaciones de transacciones similares. Los pasos son:

Identificar las transacciones del cliente.
Calcular la similitud promedio de cada transacción.
Seleccionar y ordenar las recomendaciones más relevantes.
8. Evaluación del Modelo
El modelo se evalúa utilizando el Root Mean Squared Error (RMSE) comparando las recomendaciones con los valores de transacción reales. La función evaluate_model divide el conjunto de datos y calcula el RMSE en el conjunto de prueba.

Ejecución del Proyecto
Carga los datos de transacciones en ../transaction-data/sample_dataset.csv.
Ejecuta el script para procesar, analizar y visualizar los datos.
Utiliza get_recommendations para obtener recomendaciones y evaluate_model para evaluar el desempeño del sistema.
```
