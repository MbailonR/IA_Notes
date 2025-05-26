import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

RANDOM_STATE = 42


cat_pipeline = make_pipeline( # Pipeline para características categóricas
    SimpleImputer(strategy="most_frequent"),  # Imputa valores faltantes con el valor más frecuente
    OneHotEncoder(handle_unknown="ignore")    # Codifica características categóricas
)

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Transformador que calcula la similitud entre cada instancia y los centroides de clusters
    utilizando un kernel RBF (Radial Basis Function).
    """

    # n_clusters es el numero de barrios en los que voy a dividir las zonas geográficas
    # gamma es el parámetros que controla la transicion entre puntos cercanos y lejanos, como si justase el foco de una lente
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters  # Número de clusters
        self.gamma = gamma            # Ancho de banda del kernel RBF
        self.random_state = random_state

    # Métedo fit donde vamos a entrenar usando el KMeans. Aqui es donde creamos el objeto Kmeans. Le indicamos cuantos grupos queremos, cuantas veces debe encontrar la mejor agrupación (n_init=10) y la semilla
    # Llamamos al método fit con X, que en este caso serían coordenadas de latitud y longitud
    # Con sample_weight le decimos si hay algún barrio que tiene más importancia que otro, en este caos no lo hacemos    
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10, 
                             random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
    
    # Este método toma datos (coordenada en este caso) y los transforma en nuevas características
    # En este caso, la transformación consiste en calcular la similitud de cada punto con los centros de los clusters 
    # La similitud se calcula como:  exp(-gamma * ||x - y||²), donde ||x - y|| es la distancia entre los puntos.
    # El resultado es una matriz donde cada fila corresponde a un punto en x y cada columna representa la similitud con uno de los clusters
    # Un valor cercano a 1 == Este punto está muy cerca del clúster
    # Un valor cercano a 0 == Este punto está muy lejos del clúster     
    def transform(self, X):
        # Calcula la similitud RBF entre cada instancia y los centroides de clusters
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    # Este metodo devuelve los nombres de las nuevas características generadas por el transformador.
    # En este caso serían "Cluster 0 similarity", "Cluster 1 similarity", etc. 
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
"""
Imagina que tenemos un mapa de California y queremos agrupar las casas por su ubicación geográfica. Después queremos saber qué tan cercana está cada casa a cada uno de estos grupos o "barrios". Si unas casas están en el mismo barrio lo más probable es que tenga precios similares

Usamos para esto un KMeans. Dentro del KMeans lo que ocurre es lo siguiente
 1- Elige n_clusters puntos aleatorios como centroides
 2- Asigna cada punto del dataset al centro más cercano
 3- Recalcula los centros como el promedio de todos los puntos asignados a ese grupo
 4- Repite hasta que se estabilicen los centros
 5- Una vez el KMeans ha convergido, guarda los centros finales de los clusters en el atributo cluster_centers_

X son nuestros datos de entrada de latitud y de longitud

La clase TransformerMixin está ahí para poder usar un método .fit_transform(x_train). Cuando llame a un objeto Pipeline y le diga .fit() entrenará automáticamente de esta forma

self.kmeans_ se llama con guión bajo. Esto es así por convención, al ponerles el _ se distinguen de los hiperparámetros definidos. Por ejemplo self.n_clusters es un hiperparámetro y por eso no lleva guión, al contrario que self.kmeans_ que es un atributo 

EXPANSIÓN INNECESARIA DE IMPORTANCIA DE PESOS EN LA CLASE CLUSTERSIMILARITY
Si quisiese darle más importancia a unos puntos weight sobre otros en el método fit: 
# Suponiendo que X contiene las coordenadas de latitud y longitud
# Creamos un array de pesos (1 para puntos normales, mayor para puntos más importantes)
weights = np.ones(len(X))  # Inicialmente todos los puntos tienen peso 1

# Identificamos los índices de los puntos que representan barrios importantes
important_indices = [10, 25, 50, 100]  # Por ejemplo, estos son los índices de los barrios importantes
weights[important_indices] = 5  # Les asignamos un peso mayor (por ejemplo, 5 veces más importante)

# Ajustamos el modelo con estos pesos
cluster_simil.fit(X, sample_weight=weights)
""" 




"""
Definimos varias pipelines de transformación para preparar variables numéricas
¿Para que sirve esto? para preparar datos y estandarizarlos, para crear variables nuevas y útiles como el precio por metro cuadrado gracias a precio y superficie y a transformar los datos para manejar distribuciones segadas

Para un ejemplo explicativo, digamos que tenemos lo siguiente: 
Datos: [precio, superficie] 
X = [[100000, 50],   # Precio: 100,000€, Superficie: 50m² 
     [200000, 75],   # Precio: 200,000€, Superficie: 75m²
     [np.nan, 100],  # Precio: desconocido, Superficie: 100m²
     [400000, 120]]  # Precio: 400,000€, Superficie: 120m²

Si a esto le aplico ratio_pipeline: 
   SimpleInputer va a añadir al valor nan restante la mediana, 200000. 
   FunctionTransformer calcularía el ratio precio/superficie:
[100000/50 = 2000€/m²,
 200000/75 = 2667€/m²,
 200000/100 = 2000€/m²,  # Usamos la mediana para el valor faltante
 400000/120 = 3333€/m²]
   StandardScaler estandarizaría esos valores para que tengan media 0 y desviación estándar 1

Si le aplico log_pipeline:
   SimpleInputer hace lo suyo y tenemos 200000
   FunctionTransformer calcularía el logaritmo natural de cada valor
   StandardScaler estandariza

Si aplico default_num_pipeline:
   SimpleInputer y StandardScaler hacen lo suyo
"""

# Recibe matriz X (que deberían ser datos numéricos)
# Toma la primera columna (x[:,[0]]) y la divide por la segunda (x[:,[1]]) 
# Devuelve el resultado de esa división 
def column_ratio(X):
    """Calcula el ratio entre la primera y la segunda columna"""
    return X[:, [0]] / X[:, [1]]

# Funcion para nombrar la columna resultante de la función anterior
# El nombre de la columna resultante es "ratio"
# Se utiliza con feature_names_out para que la nueva caraterística tenga un nombre adecuado 
def ratio_name(function_transformer, feature_names_in):
    """Función para nombrar las columnas de salida del ratio"""
    return ["ratio"]

# Crea una pipeline para tres operaciones secunciales
# Primero, imputa los valores faltantes NaN con la mediana
# Luego aplica la función column_ratio definida antes + ratio_name para nombrar a la columna resultante
# Finalmente, estandariza los datos para que tengan media 0 y desviación estándar 1 usando StandardScaler  
def ratio_pipeline():
    """Pipeline que crea nuevas características dividiendo dos columnas"""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

# Transformación logarítmica
# Remplaza valores NaN con la mediana 
# Aplica la función logarítmica a los datos (np.log). Con feature_names le da los mismos nombres a las columnas de salida que a las de entrada 
# Finalmente, estandariza los datos para que tengan media 0 y desviación estándar 1 usando StandardScaler 
log_pipeline = make_pipeline( # Pipeline para transformación logarítmica
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

default_num_pipeline = make_pipeline( # Pipeline por defecto para características numéricas
    SimpleImputer(strategy="median"),
    StandardScaler()
)

# Aqui es donde se uno todo lo anterior para preprocesar todos los datos
# Empieza con instancias ClusterSimilarity, que hace la vaina de Kmeans para grupar ubicaciones geográficas 
# Luego se crea un ColumnTransformer que aplica las pipelines anteriores a las columnas correspondientes  
def get_preprocessing_pipeline(n_clusters=76, gamma=1.0):
    """
    Devuelve un pipeline de preprocesamiento configurado para los datos de viviendas

    Args:
        n_clusters: Número de clusters para la similitud geoespacial. Se usa por defecto 
            el valor que mejores resultados dio en la búsqueda de hiperparámetros.
        gamma: Parámetro del kernel RBF
        
    Returns:
        ColumnTransformer: Pipeline de preprocesamiento completo
    """
    cluster_simil = ClusterSimilarity(n_clusters=n_clusters, gamma=gamma, random_state=RANDOM_STATE)
    
    return ColumnTransformer([
        # Ratios (nuevas características)
        # Todas las transformaciones aquí van a mantener el nombre de la columna original + "_ratio" o "log_" o "cat_" o "_geo" 
        """
        Imagina que tienes datos como:
        Casa 1: 3 dormitorios, 6 habitaciones totales
        Casa 2: 2 dormitorios, 4 habitaciones totales
        Esta línea crea una nueva característica: la proporción de dormitorios respecto al total de habitaciones.
        Casa 1: 3/6 = 0.5 (50 por ciento de las habitaciones son dormitorios)
        Casa 2: 2/4 = 0.5 (también 50%)
        
        Hace lo mismo para Habitaciones por casa (rooms_per_house) y Personas por casa (people_per_house)
        Recuerda que con esto sacamos datos más informativos de los que ya teníamos y que no estamos creando nuevas columnas, estamos transformando las que ya teníamos (las originales se transforman en la nueva columna) 
        """
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),        # ratio entre dormitorios y habitaciones
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),     # ratio entre habitaciones y hogares
        ("people_per_house", ratio_pipeline(), ["population", "households"]),     # ratio entre población y hogares
        
        # Transformación logarítmica para normalizar distribuciones sesgadas
        """
        Algunas variables como los ingresos suelen tener una distribución desequilibrada (muchas personas con ingresos bajos, pocas con ingresos muy altos). La transformación logarítmica "aplana" estos valores:
        Un ingreso de $1,000 se convierte en log(1000) ≈ 6.9
        Un ingreso de $10,000 se convierte en log(10000) ≈ 9.2
        Un ingreso de $100,000 se convierte en log(100000) ≈ 11.5
        
        Esto hace que la diferencia entre $1,000 y $10,000 (que es grande) se vea similar a la diferencia entre $10,000 y $100,000 en la escala logarítmica.
        """
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                              "households", "median_income"]),
        
        # Características geoespaciales utilizando clustering
        """
        Esta es la parte más interesante. Transforma las coordenadas (latitud, longitud) en algo más útil:
        Agrupa las casas en "barrios" (76 por defecto)
        Para cada casa, calcula qué tan cerca está de cada "barrio"
        Es como decir: "Esta casa está muy cerca del centro de San Francisco (0.9), algo cerca de Oakland (0.3), y lejos de Los Ángeles (0.01)".
        """
        ("geo", cluster_simil, ["latitude", "longitude"]),
        
        # Características categóricas
        """
        Procesa datos como "zona residencial", "zona comercial", etc., convirtiéndolos en números que el modelo puede entender mediante "one-hot encoding".
        Por ejemplo:

        "residencial" → [1, 0, 0]
        "comercial" → [0, 1, 0]
        "industrial" → [0, 0, 1]
        """
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    # Todas las restantes que vayan a default_num_pipeline y pista porqeu son numericas simples
    remainder=default_num_pipeline)  # Columnas restantes: housing_median_age

"""
scale_target hace que los precios (ESTO ES PARA LA VARIABLE OBJETIVO) tengan una media de 0 y desviación estándar de 1:

Si la media original es $500,000 y la desviación estándar es $200,000
Un precio de $700,000 se convierte en 1.0 (está una desviación estándar por encima de la media)
Un precio de $300,000 se convierte en -1.0 (está una desviación estándar por debajo de la media)

Esto ayuda a que los modelos de machine learning funcionen mejor.
"""
def scale_target(y_train, y_val, y_test):
    """
    Scale target variables using StandardScaler.
    
    Args:
        y_train: Training target values
        y_val: Validation target values  
        y_test: Test target values
        
    Returns:
        tuple: (scaled training data, scaled validation data, scaled test data, scaler)
    """
    y_scaler = StandardScaler()
    y_train_scaled_np = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_val_scaled_np = y_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled_np = y_scaler.transform(y_test.values.reshape(-1, 1))
    
    return y_train_scaled_np, y_val_scaled_np, y_test_scaled_np, y_scaler

"""
y_train_scaled_np = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
Aquí ocurren varias cosas:

1- y_train.values convierte los datos (probablemente un pandas Series) a un array de NumPy
2- .reshape(-1, 1) convierte el array 1D a un array 2D columnar (necesario porque scikit-learn espera matrices 2D)
3- fit_transform() hace dos cosas:
    fit(): Calcula la media y desviación estándar de los datos de entrenamiento
    transform(): Resta la media y divide por la desviación estándar

Es crucial que solo hagamos fit() en los datos de entrenamiento, no en los de validación o prueba, para evitar "data leakage" (filtración de información).

ES UN ESCALADOR NO UN MODELO. NO ESTA ENTRENANDO ESTA TRANSFORMANDO

y_val_scaled_np = y_scaler.transform(y_val.values.reshape(-1, 1))
y_test_scaled_np = y_scaler.transform(y_test.values.reshape(-1, 1))

Aquí solo aplicamos transform() (no fit_transform()), usando la media y desviación estándar que ya calculamos con los datos de entrenamiento. 
Esto es esencial para la integridad del modelo.

Finalmente devolvemos los datos escalados y el scaler para poder usarlo después en la predicción 
Al guardar y_scale podemos usarlo para transformar los datos de prueba o validación en el futuro a la escala original sin recalcular la media y desviación estándar
"""