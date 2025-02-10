import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from bosque_aleatorio import predecir_bosque, entrena_bosque

# Cargar el conjunto de datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
column_names = {
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our",
    "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail",
    "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses",
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp",
    "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs",
    "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85",
    "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re",
    "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
    "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_hash", "capital_run_length_average",
    "capital_run_length_longest", "capital_run_length_total", "spam"
}

# Leemos los datos
data = pd.read_csv(url, header = None, names = column_names)

# Separamos características (x) y etiquetas (y)
X = data.drop("spam", axis = 1).to_dict(orient = "records") # Convierte las listas a diccionarios
y = data["spam"].tolist() # Convierte a una lista

# Dividir en conjuntos de entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

def evaluar_bosque(bosque, X_test, y_test):
    """
    Evalua el rendimiento del bosque aleatorio.
    
    Parámetros:
    ---------------------------------------------------
    bosque: list(NodoN)
        Lista de árboles del bosque.
    x_test: list(dict)
        Conjunto de prueba (características).
    y_test: list
        Etiquetas reales del conjunto de prueba.
    
    Regresa:
    ---------------------------------------------------
    metricas: dict
        Diccionario con las métricas de evaluación.
    """
    y_pred = [predecir_bosque(bosque, instancia) for instancia in X_test]

    metricas ={
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_socre": f1_score(y_test, y_pred)
    }
    return metricas

# Configuraciones a probar
configuraciones = [
    {"M": 10, "max_profundidad": 5,  "variables_por_nodo": 5},
    {"M": 20, "max_profundidad": 10,  "variables_por_nodo": 10},
    {"M": 50, "max_profundidad": None,  "variables_por_nodo": 15},
    {"M": 100, "max_profundidad": 20,  "variables_por_nodo": 20},
]

# Probar cada configuración
resultados =[]
for config in configuraciones:
    print("Entrenando con M = {config['M']}, max_profundidad = {config['max_profundidad']}, variables_por_nodo = {config['variables_por_nodo']}")
    
    # Entrenar el bosque
    bosque = entrena_bosque(
        datos = X_train,
        target = "spam",
        clase_default = 0,
        M = config["M"],
        max_profundidad = config["max_profundidad"],
        variables_por_nodo = config["variables_por_nodo"]
    )

    # Evaluamos el bosque aleatorio
    metricas = evaluar_bosque(bosque, X_test, y_test)
    resultados.append((config, metricas))

    print(f"Metricas: {metricas}\n")

# Mostrar resultados
for config, metricas in resultados:
    print(f"Configuración: {config}")
    print(f"Metricas: {metricas}\n")
