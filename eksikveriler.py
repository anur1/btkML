# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

print('hello')

x=10
y=20,30
z=[2,23]

#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yükleme
veriler = pd.read_csv('eksikveriler.csv')
print(veriler)

boy = veriler [['boy']]
print(boy)

kilo = veriler [['kilo']]
print (kilo)

class makina: 
    uzunluk = 110
    km_litre = 10
    def dizel_calisma(self, km_litre):
        return km_litre+10

Skoda_tramway = makina()
print(Skoda_tramway.uzunluk)
print(Skoda_tramway.dizel_calisma(10))



#eksik veriler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean')

yas = veriler.iloc[:,1:4].values 
print(yas)

imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:, 1:4])
print(yas)






















        