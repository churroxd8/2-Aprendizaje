from collections import Counter
import random
from arboles_numericos import entrena_arbol

def entrena_bosque(datos, target, clase_default, M=10, max_profundidad=None, 
                   acc_nodo=1.0, min_ejemplos=0, variables_por_nodo=None):
    """
    Entrenamiento de un bosque aleatorio :D
    
    Parámetros:
    ------------------------------------------------
    datos: list(dict)
        Lista de diccionarios que representan las instancias.
    target: str
        Nombre del atributo del objetivo.
    clase_default: str
        Valor de la case por defecto.
    M: int
        Número de árboles en el bosque
    max_profundidad: int
        Profundidad máxima de cada árbol.
    acc_nodo: float
        Porcentaje de acierto mínimo a considerar un nodo como hoja.
    min_ejemplos: int
        Número mínimo de ejemplos para considerar un nodo como hoja.
    variables_por_nodo: int
        Número de variables a considerar en cada nodo para dividir.
    
    Regresa:
    ----------------------------------------------------
    bosque: list(NodoN)
        Lista de nodos raíz que representan los árboles del bosque.
    """

    bosque = []
    n = len(datos)

    # Muestreo aleatorio con repetición
    for _ in range(M):
        subconjunto = random.choices(datos, k=n)

        # Entrenamos un árbol con el subconjunto
        arbol = entrena_arbol(
            datos=subconjunto,
            target=target,
            clase_default=clase_default,
            max_profundidad=max_profundidad,
            acc_nodo=acc_nodo,
            min_ejemplos=min_ejemplos,
            variables_seleccionadas=variables_por_nodo # Número de variables por el nodo
        )

        # Agregar el árbol al bosque
        bosque.append(arbol)

    return bosque

def predecir_bosque(bosque, instancia):
    """
    Realizamos una predicción para una instancia utilizando el bosque aleatorio.
    
    Parámetros:
    -----------------------------------------------------------------------------------
    bosque: list(NodoN)
        Lista de árboles del bosque.
    instancia: dict
        Instancia para la cual se desea hacer la predicción.
    
    Regresa:
    -----------------------------------------------------------------------------------
    predicción: str
        Clase predicha por el bosque"""
    
    predicciones = []

    # Predicción con cada árbol
    for arbol in bosque:
        prediccion = predecir_arbol(arbol, instancia)
        predicciones.append(prediccion)

        # Votación mayoritaria
        return Counter(predicciones).most_common(1)[0][0]
    
def predecir_arbol(nodo, instancia):
    """"
    Realizar una predicción para una instancia utilizando un árbol de decisión.
    
    Parámetros:
    -----------------------------------------------------------------------------------
    nodo: NodoN
        Nodo raíz del árbol.
    instancia: dict
        Instancia para la cual se desea hacer la predicción.
        
    Regresa:
    -----------------------------------------------------------------------------------
    prediccion: str
        La clase predicha por el árbol.
    """

    if nodo.terminal:
        return nodo.clase_default
    
    if instancia[nodo.atributo] < nodo.valor:
        return predecir_arbol(nodo.hijo_menor, instancia)
    
    else:
        return predecir_arbol(nodo.hijo_mayor, instancia)