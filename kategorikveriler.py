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



#kategorik veriler tr->1

ulke = veriler.iloc[:, 0:1].values
print(ulke)


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:, 0] =le.fit_transform(veriler.iloc[:, 0])

print(ulke)  

#kategorik veriler tr_us_fr->100
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

print(ulke)



#verilerin birleştirilmesi (sayısal + kategorik)
sonuc = pd.DataFrame(data=ulke, index=range(22), columns = ['fr'
, 'tr', 'us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=yas, index=range(22), columns =  ['boy', 'kilo', 'yas'])
print(sonuc2)


cinsiyet = veriler.iloc[:, -1].values
sonuc3 = pd.DataFrame(data=cinsiyet, index= range(22), columns = ['cinsiyet']
)
print(sonuc3)

s = pd.concat([sonuc, sonuc2], axis=1)
print(s)

s2 = pd.concat ([s, sonuc3], axis=1)
print(s2)





#verilerin eğitim+test kümesine bölünmesi

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33, random_state=0) 












        