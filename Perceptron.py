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




entradas = np.asarray([[-1,-1],[-1,1],[1,-1],[1,1]])
linhasEntrada = len(entradas)
colunasEntrada = len(entradas[0])
saidas = np.asarray([-1, -1, -1, 1]).reshape((4,1))


dimensoes = colunasEntrada
Ni = 0.1
Wini = 0.01
Ws = np.ones((dimensoes + 1,1), float)
Ws[0] = 1
Threshold = 0

def DeltaW(target, output, Xi):
	return Ni * (target - output) * Xi


def Fx(W, X):
	soma = 0;

	for i in range(0, dimensoes):
		soma += W[i + 1] * X[i]

	return 1 if soma + W[0] >= Threshold else -1


def train(x, saida):
    output = Fx(Ws, x);

    for i in range(0, dimensoes):
        Ws[i+1] += DeltaW(saida, output, x[i])
    Ws[0] = Ni * (saidas[i] - output)
    return
    
def perceptron():
    for i in range(1,dimensoes + 1):
        Ws[i] = 1* Wini
        
    return


def transform(entrada):
    return Fx(Ws, entrada)

def showDivision(entradas):
    divisionPlane = np.mgrid[-1:1.1:0.05, -1:1.1:0.05].reshape(2,-1).T
    plt.figure()
    for i in range(0,len(divisionPlane)):
        output = transform(divisionPlane[i])
        c = 'deepskyblue' if output == 1 else 'lightcoral'
        plt.scatter(divisionPlane[i,0], divisionPlane[i,1], c = c, cmap = 'rainbow', s = 10)
    for i in range(0, linhasEntrada):
        c = 'blue' if saidas[i] == 1 else 'red'
        plt.scatter(entradas[i,0], entradas[i,1], c = c, cmap = 'rainbow', s = 20)

    plt.title('Perceptron - Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    return

#main
perceptron();

for i in range(0,1000):
    for j in range(0, linhasEntrada):
        entrada = entradas[j]
        train(entrada, saidas[j])
        
for i in range(0, linhasEntrada):
    output = transform(entradas[i])
    print('Saida pra o conjunto de entradas ', entradas[i,0], 'e', entradas[i, 1], 'é:', output)
		
showDivision(entradas)
