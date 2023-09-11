# RetroMod2

## Implementación de una técnica de aprendizaje máquina sin el uso de un framework

Este repositorio contiene una implementación básica de un clasificador de Árbol de Decisión en Python. El código implementa la construcción de un árbol de decisión y su uso para hacer predicciones en nuevos datos. El objetivo de este codigo es demostrar el funcionamiento del algoritmo y realizar pruebas con datos nuevos.

Para este caso en específico se ha utilizado el dataset Iris por su versatilidad y naturaleza para clasificación.

## Contenido

- [Requisitos](#requisitos)
- [Instrucciones de Uso](#instrucciones-de-uso)
- [Funcionamiento](#funcionamiento)
## Requisitos

- Python 3.x
- Bibliotecas: numpy, pandas (para generar datos y visualización de resultados)

## Instrucciones de Uso

1. Clona este repositorio en tu máquina local:

https://github.com/juancarloscorona14/RetroMod2.git

2. Navega al directorio del repositorio

3. Configura la ubicacion del dataset de acuerdo a la localización del archivo

4. Ejecuta

5. Si deseas cambiar los datos de prueba, por favor modifica el archivo class_arbol_decision.py

## Funcionamiento
Esta seccion contiene una descripción general de la estructura del programa.

### Clase Node (Nodo)
La clase '**Node**' describe los nodos de nuestro árbol de decisión. Contiene los siguientes atributos:

- *'feature_index'*: Índice de la característica utilizada para dividir.

- *'threshold'*: Valor de umbral para la característica.

- *'left'*: Referencia al nodo hijo izquierdo.

- *'right'*: Referencia al nodo hijo derecho.

- *'information_gain'*: Ganancia de información obtenida por la división en este nodo.

- *'value'*: Para los nodos hoja, almacena la etiqueta de la clase mayoritaria.

### Clase Tree (Árbol)
La clase '**Tree**' implementa el algoritmo del árbol de decisión. Incluye los siguientes métodos:

1. Constructor '**_ _init_ _(_self, minimum_sample_split = 2, max_depth = 2)_**: Inicializa el árbol de decisión con parámetros para las condiciones de detención.

- **'root'**: Nodo raíz del árbol.

- **'minimum_sample_split'**: Número mínimo de muestras requeridas para continuar dividiendo un nodo.

- **'max_depth'**: Profundidad máxima del árbol.

2. **build_tree(_self, x, y, current_depth_)**

Esta es una función recursiva para construir el árbol binario. Utiliza los siguientes pasos:

    1. Comprueba si se cumplen las condiciones de detención (muestras mínimas, profundidad máxima).

    2. Encuentra la mejor división usando la función get_best_split.

    3. Si la ganancia de información > 0, crea subárboles izquierdo y derecho de forma recursiva.

    4. Retorna un nodo de decisión con atributos.

Métodos utilizados:

- **get_best_split(self, dataset, num_samples, num_features)**

    Encuentra la mejor división para un índice de característica dado. Retorna un diccionario que contiene información sobre la mejor división. 

- **split(self, dataset, feature_index, threshold)**

    Divide el conjunto de datos basado en el índice de característica y el umbral dados. Retorna subconjuntos izquierdos y derechos.

- **information_gain(self, parent, l_child, r_child, mode="entropy")**

    Calcula la ganancia de información utilizando el índice Gini o Entroía (default).

- **entropy(self, y)**

    Calcula la entropía de una variable objetivo.

- **gini_index(self, y)**

    Calcula el índice Gini de una variable objetivo.

- **calculate_leaf_value(self, Y)**

    Encuentra la etiqueta de clase mayoritaria para un nodo hoja.

- **make_prediction(self, x, tree)**

    Predice la etiqueta de una sola instancia de datos.

- **fit(self, X, Y)**

    Ajusta el árbol de decisión a los datos de entrenamiento.

- **predict(self, X)**

    Predice las etiquetas para un nuevo conjunto de datos utilizando el árbol de decisión entrenado.
    
- **print_tree(self, tree=None, indent=" ")**

    Imprime la estructura del árbol de decisión.







## Referencias

- https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/resources/iris/
  
- https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/

