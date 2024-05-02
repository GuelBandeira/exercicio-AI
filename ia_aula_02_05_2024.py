import pandas as pd 
# Lib de computação científica: 
import numpy as np  

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
#new_names = ['sepal_length','sepal_width','petal_length','petal_width','iris_class']
new_names = ['altura_sépala','largula_sépala','altura_pétala','largura_pétala','tipo_de_iris']
dataset = pd.read_csv(url, names=new_names, skiprows=0, delimiter=',')
dataset.info()

dataset.head(150)

y = dataset['tipo_de_iris']
# Lembramos de remover a saída dos x's de entrada: 
x = dataset.drop(['tipo_de_iris'], axis=1)

print ("dataset : ",dataset.shape)
print ("x : ",x.shape)
print ("y : ",y.shape)

y=pd.get_dummies(y)
# Imprimindo 9 exemplos aleatoriamente, para ver os dados:
y.sample(9)
# Agora que nosso dataset está pronto, temos que separá-lo em conjunto de treinamento e teste. 
# Podemos fazer isso usando o método scikit learn train_test_split():

# Importa o train_test_split do Scikit Learn: 
from sklearn.model_selection import train_test_split

# Gerando conjunto de treino e teste (15% para teste):
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15) #0.15 data as data test

# Precisamos converter nosso conjunto de dados para float 32bits, que é o que a MLP recebe:
x_train = np.array(x_train).astype(np.float32)
x_test  = np.array(x_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test  = np.array(y_test).astype(np.float32)

# Vamos imprimir o conjunto de dados para validação:
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Nossa rede neural será 4 x 3 x 2 x 3, portanto as camadas ocultas são (3 unidades e 2 unidades). 
# Definimos nossa iteração máxima como 5.000 para treinar na época de 5.000 e 
# alfa como 0,01 para nossa taxa de aprendizado. Definimos verbose como 1 imprimir as saídas 
# durante o processo de treinamento. Random_state é usado como uma semente aleatória:

# Importanto a lib de MLP
from sklearn.neural_network import MLPClassifier

# Initialização do modelo:
Model = MLPClassifier(hidden_layer_sizes=(3,2), max_iter=10000, alpha=0.01, #try change hidden layer
                     solver='sgd', verbose=1,  random_state=233) #try verbode=0 to train with out logging
#train our model
h=Model.fit(x_train,y_train)

#use our model to predict
y_pred=Model.predict(x_test)

# Depois de terminar o processo de treinamento, podemos usar nosso modelo treinado 
# com o método model.predict(). Para obter o nosso resultado de classificação, podemos 
# importar classification_report de sklearn.matrix e chamar 
# classification_report( "saída correta, predição" ). Para mostrar os resultados 
# na matriz de confusão e precisão, também precisamos importá-los de sklearn.matrix:

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Imprime relatório de performance:
print(classification_report(y_test,y_pred))
# Imprime a matriz de confusão (Confusion matrix): 
print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
# Imprime a taxa de acerto:  
print('accuracy is ',accuracy_score(y_pred,y_test)) 

import matplotlib.pyplot as plt

plt.plot(h.loss_curve_)
plt.title('Loss History')
plt.xlabel('epoch')
plt.legend(['Loss'])


# Lembrando de como codificamos nossas saídas:

# Iris setosa: 100
Iris_setosa = np.array( [[1,0,0]] )
# Iris versicolor: 010 
Iris_versicolor = np.array( [[0,1,0]] )
# Iris virginica: 001
Iris_virginica  = np.array( [[0,0,1]] )

# Agora podemos fazer previsões com o nosso modelo:
entrada_exemplo = np.array([[5.4,3.7,1.5,0.2]])

y_previsto = Model.predict(entrada_exemplo)

# Caso a saída seja Iris setosa (100):
if np.array_equal(y_previsto, Iris_setosa) == True:
  print('Iris setosa')

# Caso a saída seja Iris versicolor (010):
if np.array_equal(y_previsto, Iris_versicolor) == True:
  print('Iris versicolor')

# Caso a saída seja Iris virginica (001):
if np.array_equal(y_previsto, Iris_virginica) == True:
  print('Iris virginica')

