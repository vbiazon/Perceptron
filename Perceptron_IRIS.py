# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 23:41:32 2019

@author: Victor Biazon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
import random
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

def Encoder(data): #função para codificar os dados em strings para numeros para poderem ser calculados no perceptron
    
    enc = LabelEncoder()
    label_encoder = enc.fit(data)
    y = label_encoder.transform(data) + 1

    return y

def DeltaW(target, output, Xi): #calcula o Delta W sendo esta a correção aplicado aso pesos
	return Ni * (target - output) * Xi


def Fx(W, X): #calcula o output, aplicando as entradas aos pesos e passando pelo função de transferencia
	soma = 0;

	for i in range(0, len(X)):
		soma += W[i + 1] * X[i]

	return 1 if soma + W[0] >= Threshold else -1


def train(x, saida): #recalcula os pesos baseados nas saidas resultantes dos pesos e relação as saidas conhecidas
    output = Fx(Ws, x);

    for i in range(0, dimensoes):
        Ws[i+1] += DeltaW(saida, output, x[i])
    Ws[0] = Ni * (saidas[i] - output)
    return
    
def perceptron(): #inicializa os pesos com valores bem pequenos
    for i in range(1,dimensoes + 1):
        Ws[i] = 1* Wini
        
    return


def transform(entrada): #realiza o calculo da saida a partir de uma entrada
    return Fx(Ws, entrada)

def showDivision(entradas): #plota os pontos de acordo com a classificação do perceptron
    divisionPlane = np.mgrid[4:7.5:0.07, 1:5:0.07].reshape(2,-1).T
    plt.figure()
#    for i in range(0,len(divisionPlane)):
#        output = transform(divisionPlane[i])
#        c = 'deepskyblue' if output == 1 else 'lightcoral'
#        plt.scatter(divisionPlane[i,0], divisionPlane[i,1], c = c, cmap = 'rainbow', s = 10)
    for i in range(0, linhasEntrada):
        c = 'blue' if saidas[i] == 1 else 'red'
        plt.scatter(entradas[i,0], entradas[i,1], c = c, cmap = 'rainbow', s = 20)

    plt.title('Perceptron - Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    
    return

#main
#le dados do dataset 
data = pd.read_table('Iris.txt', decimal  = ",")

x = np.asarray(data.iloc[:,:-1]) #separa dados em variaveis independentes e a saida
y = Encoder(np.asarray(data.iloc[:,-1]))    
x_l = []
y_l = []
for i in range(0,len(x)):
    if y[i] == 1 or y[i] == 2:
        x_l.append(x[i,:])
        y_l.append(y[i])

for i in range(0, len(y_l)):
    if y_l[i] == 2:
        y_l[i]= -1
        
x = np.asarray(x_l)
y = np.asarray(y_l)
del x_l 
del y_l



entradas = np.copy(x)
linhasEntrada = len(entradas)
colunasEntrada = len(entradas[0])
saidas = np.copy(y)

#parametros do perceptron
dimensoes = colunasEntrada
Ni = 0.1
Wini = 0.01
Ws = np.ones((dimensoes + 1,1), float)
Ws[0] = 1
Threshold = 0

perceptron();
#realiza treinamento do perceptron
for i in range(0,1000):
    for j in range(0, linhasEntrada):
        entrada = entradas[j]
        train(entrada, saidas[j])
        
    
#realiza teste para verificação da classificação do perceptron      
y_pred = []
for i in range(0, linhasEntrada):
    output = transform(entradas[i])
    y_pred.append(output)
    print('Saida pra o conjunto de entradas ', entradas[i,0], 'e', entradas[i, 1], 'é:', output)
		
showDivision(entradas)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)