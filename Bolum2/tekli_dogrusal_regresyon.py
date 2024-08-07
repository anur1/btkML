# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)


#8 - model inşası (lineer regresyon)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train, Y_train)
tahmin = lr.predict(X_test)


#non-standard tahmin
lr.fit(x_train, y_train)
tahmin2 = lr.predict(x_test)


#veri görselleştirme
#sıralama
x_train = x_train.sort_index()
y_train = y_train.sort_index()
#çizdirme
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
#etiketleme
plt.title("aylara göre satış")
plt.xlabel("aylar")
plt.ylabel("satışlar")


