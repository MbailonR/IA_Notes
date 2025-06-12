from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

class AgeOutlierImputer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para reemplazar edades mayores a un umbral (por defecto, 150)
    con la mediana del resto de edades válidas.
    """
    def __init__(self, threshold=150):
        self.threshold = threshold  # Umbral considerado como outlier
        self.median_age = None      # Aquí se guardará la mediana
    
    def fit(self, X, y=None):
        # Calculamos la mediana de las edades menores o iguales al umbral
        self.median_age = np.median(X[X <= self.threshold])
        return self  # Necesario para que funcione en pipelines
    
    def transform(self, X):
        X_copy = X.copy()                # Copiamos los datos para no modificar el original
        mask = X_copy > self.threshold   # Creamos una máscara booleana para los outliers
        X_copy[mask] = self.median_age   # Reemplazamos outliers por la mediana
        return X_copy                    # Devolvemos el resultado


class TSHLogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para aplicar logaritmo a TSH, ya que tiene distribución sesgada.
    """
    def fit(self, X, y=None):
        return self  # No necesita ajuste previo
    
    def transform(self, X):
        X_copy = X.copy()
        # Usamos log1p para manejar ceros (log1p(x) = log(1 + x), siempre definido)
        return np.log1p(X_copy)
    
# Variables numéricas "normales"
standard_numeric = ['TT4', 'T4U', 'FTI', 'T3']         

# TSH se trata aparte (por su distribución)
tsh_feature = ['TSH']                                  

# Variables categóricas binarias
binary_features = [                                    
    'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
    'query_hypothyroid', 'query_hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych'
]

# Variables categóricas con más de 2 valores
categorical_features = ['sex', 'referral_source']      

# Create specialized pipelines
age_pipeline = Pipeline([
    ('outlier_handler', AgeOutlierImputer(threshold=150)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

standard_numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

tsh_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log_transform', TSHLogTransformer()),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])


def get_preprocessor():
    return ColumnTransformer(
    transformers=[
        ('age', age_pipeline, ['age']),
        ('standard_numeric', standard_numeric_pipeline, standard_numeric),
        ('tsh', tsh_pipeline, tsh_feature),
        ('binary', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), 
            binary_features),
        ('categorical', categorical_pipeline, categorical_features),
    ],
    remainder='drop'
)
