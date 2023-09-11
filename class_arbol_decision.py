""" Implementación de una técnica de aprendizaje máquina sin el uso de un framework """
""" Juan Carlos Corona Vega A01660135 """

# Importando librerias
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder # Para relizar encoding de la variable objetivo
from sklearn.model_selection import train_test_split # Para separar en entrenamiento y prueba
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score # Para evaluar el modelo

# Implementacion del algoritmo de arbol de decision sin framework
""" Definición de métodos del algoritmo utilizando POO"""
"""-----------------------------------------------------------------------------------------------"""
print('Inicializando...\n')

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor que representa a los nodos de decision ''' 
        
        # nodo de decision 
        self.feature_index = feature_index # indice de cada feature del dataset
        self.threshold = threshold # umbral para ramificar
        self.left = left # subarboles izquierdos
        self.right = right # subarboles derechos
        self.info_gain = info_gain # indice de ganancia de informacion
        
        # nodo hoja
        self.value = value

# clase arbol
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # inicializar la raíz del árbol para tener una referencia al momento de desplazarnos por el arbol
        self.root = None
        
        # limitantes
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' funcion recursiva para construir el arbol a traves de la division
        de los datos tomando en cuenta la ganancia de informacion en cada division''' 
        
        ''' Es recursiva porque estamos estamos llamando a la función build_tree dentro de la misma función
        por lo que primero creará todos los subárboles de la izquierda y luego una vez que haya alcanzado 
        el nodo hoja creará los subárboles de la derecha'''

        # Division entre features y target
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # dividir si es que no se han cumplido las condiciones de parada
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            
            # hallar la division optima con metodo get_best_split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            
            # comprobar si la ganancia de información es diferente de cero para la izquierda y derecha
            # una ganancia de información igual a cero significa que estamos dividiendo un nodo que ya es puro
            if best_split["info_gain"] > 0:
                # sub arboles por la izquierda
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)  # se incrementa el valor curr_dept para verificar que no hemos excedido la profundidad
                # sub arboles por la derecha
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)

                # regresar nodo de decision con 
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # calcular nodo de hoja
        leaf_value = self.calculate_leaf_value(Y)
        # regresar nodo de hoja
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' función para encontrar la mejor division en los datos y regresa un diccionario
        con '''
        
        # en este diccionario se almacenan las divisiones y se toma la mejor
        best_split = {}
        # se inicializa en negativo porque queremos maximizar la ganancia de informacion
        max_info_gain = -float("inf")
        
        # iterar para todas las columnas (features)
        for feature_index in range(num_features):

            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values) # obtener solo los valores unicos que contiene cada feature
            
            # iterar sobre los valores unicos  de las columnas
            for threshold in possible_thresholds:
                # obtener la division actual
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                
                # condicion para verificar que la ramificacion no sea cero (child =! 0)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    # Utilizando unicamente target values
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    
                    # calcular ganancia de infromacion actual utilizando gini
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    
                    # actualiza los valores si se encuentra un mejor division
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        
        print(f'Ganancia de Informacion: {max_info_gain}')

        # regresa la mejor division
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' dividir un conjunto de datos en dos subconjuntos en función de un índice de características y un valor umbral determinados. '''
        
        # la division (izquierda o derecha) se basa en el valor del threshold (umbral)
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold]) # contiene las filas en las que el valor del feature es **inferior o igual** al threshold
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold]) # contiene las filas en las que el valor del feature es **superior** al threshold
        
        return dataset_left, dataset_right
        # Estos dos subconjuntos representan los datos que se enviarán a los nodos hijos izquierdo y derecho del arbol, respectivamente
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' calcula la ganancia de información en función del modo especificado ("entropy" o "gini") '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        
        if mode == " gini ":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
        # regresa la medida (gain) de cuánta incertidumbre o impureza se reduce al dividir un conjunto de datos en nodos hijos.

    def entropy(self, y):
        ''' función para calcular la entropia '''
        # Revisar el apartado de referencias en la documentacion para más detalles en el funcionamiento de la formula
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        ''' función para calcular el índice de gini '''
        # Revisar el apartado de referencias en la documentacion para más detalles en el funcionamiento de la formula
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        ''' funcion para calcular el nodo hoja '''
        # un nodo hoja es la clase mayoritaria presente en ese nodo concreto
        # encontrar el elemento más presente en y
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        ''' funcion simple para visualizar el arbol de decision mostrandose los niveles de este'''
        
        # checar que el arbol no sea nulo
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+ str(tree.feature_index), " <= ", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

            # Se imprime la estructura del arbol mostrando en cada nodo el indice de la variable, el umbral y la ganancia de informacion
    
    def fit(self, X, Y):
        ''' funcion para entrenar el arbol '''
        # se encarga de unir las features con la variable objetivo y manda llamar el método build_tree
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' funcion para predecir un dataset unicamente con columnas de features'''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' realiza predicciones para un punto de datos de entrada dado x recorriendo el árbol de decisión. '''
        
        # checar si el arbol actual es un nodo hoja, si no es ninguno, regresa el valor como predicción, indicando que el árbol ha alcanzado un nodo hoja
        if tree.value!=None: return tree.value
        
        # extraer el valor de los features para compararlos con el threshold y decidir si se irán por la izquierda o derecha
        feature_val = x[tree.feature_index]
        
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
        ''' Este proceso continúa hasta que se alcanza un nodo hoja, y se devuelve la predicción basada en la clase mayoritaria almacenada en el nodo hoja.'''


        
""" Construccion, Entrenamiento y Prueba del Modelo """
"""-----------------------------------------------------------------------------------------------"""

print('Cargando Datos...\n')
data = pd.read_csv("iris.csv", index_col= False)
data.info()

# Preprocesamiento de datos
label_encoder = LabelEncoder()
data["type"] = label_encoder.fit_transform(data["type"])
print('\nVista previa de los datos a utilizar:\n\n')
print(data)

# Separacion de los datos en entrenamiento y prueba utilizando train_test_split de sklearn
print('\nSeparando en conjuntos de entrenamiento...\n')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20)

print('Inicializando modelo...\n')

best_accuracy = 0.0
best_min_samples_split = 0
best_max_depth = 0

for min_sample in range(2,9):
    for max_depth1 in range(2,5):
        classifier = DecisionTreeClassifier(min_samples_split = min_sample, max_depth = max_depth1)
        classifier.fit(X_train,Y_train)
        Y_pred = classifier.predict(X_test)
        score = accuracy_score(Y_test, Y_pred)
        print(f'\nMínimo de muestras para dividir un nodo: {classifier.min_samples_split}\nProfundidad máxima del árbol: {classifier.max_depth}\n')

        # Condiciones para determinar mejores parametros
        if score > best_accuracy:
            best_accuracy = score
            best_min_samples_split = min_sample
            best_max_depth = max_depth1

print('Mejores Resultados del entrenamiento:')
print(f'Mínimo de muestras para dividir un nodo: {best_min_samples_split}\nProfundidad máxima del árbol: {best_max_depth}\nScore: {best_accuracy}')

print('\nNOTA: Referirse a la documentacion para conocer mas detalles sobre la interpretacion\nde la ganancia de información para cada iteracion de parametros\n')

print('\nImprimiendo arbol...\n')
classifier.print_tree()


print('\nEvaluacion del modelo:\n\n')

# Evaluacion del modelo con diferentes métricas
Y_pred = classifier.predict(X_test)

# Precision
precision = precision_score(Y_test, Y_pred, average='weighted')

# Recall
recall = recall_score(Y_test,Y_pred, average='weighted')

# F1 score
f1 = f1_score(Y_test, Y_pred, average='weighted')

# Matriz de confusion
confusion = confusion_matrix(Y_test, Y_pred)
print(f'Accuracy (Exactitud): {best_accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Matriz de confusion:\n')
print(confusion)


# hacer predicciones unicamente con el mejor modelo
print('\n\nPredicciones:\n\n')
# Datos de prueba
# Conjunto de 4 valores representa las 4 caracteristicas del dataset
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X_prueba = [[4.5, 3.2, 1.3, 0.2],  # Ejemplo 1
          [6.7, 3.1, 4.4, 1.4],  # Ejemplo 2
          [5.1, 2.9, 3.3, 1.0],  # Ejemplo 3
          [5.0, 2.0, 3.5, 1.0],  # Ejemplo 4
          [5.9, 3.0, 5.1, 1.8]]  # Ejemplo 4
pred_df = pd.DataFrame(X_prueba, columns=col_names)
print('Datos de prueba:')
print(pred_df)

# Hacer predicciones para cada ejemplo en X_prueba
predictions = [classifier.make_prediction(x, classifier.root) for x in X_prueba]
print('\nPredicciones:')
# Imprimir las predicciones
pred_df['predicted_type'] = predictions
print(pred_df)
print('\n\nInterpretación del encoding\n 0 = Iris-setosa \n 1 = Iris-versicolor \n 2 = Iris-virginica')
