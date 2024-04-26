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