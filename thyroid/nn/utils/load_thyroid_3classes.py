import pandas as pd
from sklearn.model_selection import train_test_split


def load_thyroid_3classes():
    # Ruta al archivo CSV que contiene los datos del problema de tiroides
    file_path = "../data/thyroidDF.csv"
    
    # Cargamos el archivo CSV en un DataFrame de pandas
    df = pd.read_csv(file_path)

    # Diccionario que mapea las etiquetas originales a tres clases simplificadas
    class_mapping = {
        '-': 'negative',
        'K': 'hyperthyroid', 'B': 'hyperthyroid', 'H|K': 'hyperthyroid',
        'KJ': 'hyperthyroid', 'GI': 'hyperthyroid',
        'G': 'hypothyroid', 'I': 'hypothyroid', 'F': 'hypothyroid', 'C|I': 'hypothyroid',
        'E': 'negative', 'LJ': 'negative', 'D|R': 'negative',
    }

    # Aplicamos el mapeo anterior a la columna 'target' para transformar sus valores
    df['target'] = df['target'].map(class_mapping)

    # Eliminamos las filas donde 'target' es NaN (es decir, no se pudo mapear correctamente)
    df = df.dropna(subset=['target'])

    # Buscamos todas las columnas cuyo nombre termina con '_measured'
    measured_cols = [col for col in df.columns if col.endswith('_measured')]

    # Definimos una lista de columnas que vamos a eliminar:
    # - 'patient_id': probablemente es solo un identificador, no útil para entrenar el modelo
    # - columnas '_measured': indican si una variable fue medida, no el valor en sí
    # - 'TBG': demasiados valores faltantes (~96.6%)
    columns_to_drop = [
        'patient_id',
        *measured_cols,  # Agregamos todas las columnas '_measured'
        'TBG',
    ]

    # Creamos el conjunto de características (X) eliminando las columnas no deseadas y la columna 'target'
    X = df.drop(columns_to_drop + ['target'], axis=1)

    # El objetivo (y) será la columna 'target' ya simplificada
    y = df['target']

    # Dividimos los datos en entrenamiento y prueba:
    # - 80% para entrenamiento, 20% para prueba
    # - usamos random_state=42 para que la división sea siempre igual
    # - stratify=y asegura que se mantenga la proporción de clases en ambos conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Devolvemos los datos divididos
    return X_train, X_test, y_train, y_test