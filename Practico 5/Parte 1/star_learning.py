
#%% Importar librerias 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Para mostrar el arbol de decision
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
from IPython.display import Image  
import pydotplus

import os

#%% Leer csv Titanic
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'estrellas.csv')
#print("Path: ",csv_path)
star_data = pd.read_csv(csv_path)
print(star_data.head())

#%% Evaluar datos faltantes
print(star_data.isnull().sum())

# %% Separar en train y test
X = star_data.drop(columns='Spectral Class')
y = star_data['Spectral Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Dividir train en train y validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# %% Preprocesar datos faltantes
# No hay datos faltantes
#%% Preprocesar variables categoricas
def preprocesar_variables_categoricas(data):
    '''Preprocesar variables categoricas'''
    # Obtener dummies de columnas Star type, Star category, Star color
    dummies = pd.get_dummies(data[['Star type', 'Star category', 'Star color']])
    print(dummies)
    # Eliminar columnas originales
    data.drop(columns=['Star type', 'Star category', 'Star color'], inplace=True)
    # Concatenar dummies
    data = pd.concat([data, dummies], axis=1)
    return data

X_train = preprocesar_variables_categoricas(X_train)
X_val = preprocesar_variables_categoricas(X_val)
X_test = preprocesar_variables_categoricas(X_test)

#%% Eliminar columnas innecesarias
# No hay columnas innecesarias

features = list(X_train.columns)
#%% Normalizar datos
def normalizar_datos(data):
    '''Normalizar datos'''
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

X_train = normalizar_datos(X_train)
X_val = normalizar_datos(X_val)
X_test = normalizar_datos(X_test)

# %% Entrenar modelo DecisionTree

# Check the shape of X_train and X_val
assert X_train.shape[1] == X_val.shape[1], "Mismatch in number of features between X_train and X_val"

modelo_decision_tree = DecisionTreeClassifier()
modelo_decision_tree.fit(X_train, y_train)
y_pred = modelo_decision_tree.predict(X_val)

# %% Mostrar arbol de decision
dot_data = StringIO()
export_graphviz(modelo_decision_tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('arbol_decision.png')
Image(graph.create_png())

#%% Evaluar accuracy, precision, recall y F1
def obtener_metricas_evaluacion(y_real, y_predicho):
    '''Obtener metricas de evaluacion'''
    print('Accuracy:', accuracy_score(y_real, y_predicho))
    print('Precision:', precision_score(y_real, y_predicho))
    print('Recall:', recall_score(y_real, y_predicho))
    print('F1:', f1_score(y_real, y_predicho))
    
    matrix_confusion = confusion_matrix(y_real, y_predicho)
    sns.heatmap(matrix_confusion, annot=True, fmt='d')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.show()
    
obtener_metricas_evaluacion(y_val, y_pred)

# %% Entrenar modelo Random Forest
modelo_random_forest = RandomForestClassifier()
modelo_random_forest.fit(X_train, y_train)
y_pred = modelo_random_forest.predict(X_val)

obtener_metricas_evaluacion(y_val, y_pred)

# %% Entrenar modelo KNN
modelo_knn = KNeighborsClassifier(n_neighbors=3)
modelo_knn.fit(X_train, y_train)
y_pred = modelo_knn.predict(X_val)

obtener_metricas_evaluacion(y_val, y_pred)

# %% Entrenar modelo Naive Bayes
modelo_naive_bayes = GaussianNB()
modelo_naive_bayes.fit(X_train, y_train)
y_pred = modelo_naive_bayes.predict(X_val)

obtener_metricas_evaluacion(y_val, y_pred)
# %% Entrenar modelo SVM
modelo_svm = SVC()
modelo_svm.fit(X_train, y_train)
y_pred = modelo_svm.predict(X_val)

obtener_metricas_evaluacion(y_val, y_pred)

# %% Entrenar modelo Regresion Logistica
modelo_regresion_logistica = LogisticRegression()
modelo_regresion_logistica.fit(X_train, y_train)
y_pred = modelo_regresion_logistica.predict(X_val)

obtener_metricas_evaluacion(y_val, y_pred)
# %%
