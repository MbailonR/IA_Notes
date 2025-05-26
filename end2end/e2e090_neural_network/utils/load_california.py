import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def load_housing_data(filepath="../data/housing.csv",
                      test_size=0.2,
                      random_state=RANDOM_STATE) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    # El tuple de arriba indica que la función devolverá una tupla con 4 series de pandas
    housing = pd.read_csv(filepath)
    # Aquí creamos una nueva columna "income_cat" que categoriza los ingresos en 5 categorías
    # 0-1.5, 1.5-3.0, 3.0-4.5, 4.5-6.0 y >6.0 
    # Esto es útil para realizar un muestreo estratificado 
    # labels=[1, 2, 3, 4, 5] significa que la primera categoría será 1, la segunda 2, etc. 
    housing["income_cat"] = pd.cut(housing["median_income"],
                                  bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                  labels=[1, 2, 3, 4, 5])
    # Dividimos el conjunto de datos en un conjunto de entrenamiento y otro de prueba (80/20)
    # stratify=housing["income_cat"] significa que ambos ocnjuntos tendrán la misma proporción de ingresos, para que tengamos una distribución fiel de los datos
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=test_size, stratify=housing["income_cat"], random_state=random_state)
    
    # Ahora que estratificamos ya no necesitamos income_cat, así que nos la cargamos
    # axis=1 significa que queremos eliminar una columna, y inplace=True significa que queremos modificar el DataFrame original 
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    
    X_train = strat_train_set.drop("median_house_value", axis=1) # Características para entrenar el modelo, eliminamos la variable objetivo 
    y_train = strat_train_set["median_house_value"].copy() # Valor objetivo para entrenar 
    
    X_test = strat_test_set.drop("median_house_value", axis=1) # Características para evaluar el modelo, eliminamos la variable objetivo 
    y_test = strat_test_set["median_house_value"].copy() # Valor objetivo para evaluar 
    
    
    return X_train, X_test, y_train, y_test 
