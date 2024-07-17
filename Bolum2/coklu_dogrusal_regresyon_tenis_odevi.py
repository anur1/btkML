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
veriler = pd.read_excel('odev_tenis.xlsx')
print(veriler)
#veri on isleme



#2.2 data görme
# aylar = veriler[['Aylar']]
# print(aylar)
# satislar = veriler[['Satislar']]
# print(satislar)
# satislar2 = veriler.iloc[:,:1].values
# print(satislar2)



#kategorik veriyi sayısal 1-0'a çevirme
from sklearn import preprocessing
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform) #bütün sütunları toptan encode eder
print(veriler2)
c = veriler2.iloc[:, :1]
from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data = c, index=range(14), columns=['overcast', 'rainy', 'sunny'])
sonveriler = pd.concat([havadurumu, veriler.iloc[:, 1:3]], axis=1 ) #havadurumu 
sonveriler = pd.concat([veriler2.iloc[:, -2:], sonveriler], axis=1)


# #verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:, :-1],sonveriler.iloc[:, -1:],test_size=0.33, random_state=0)

# #verilerin olceklenmesi
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)
# Y_train = sc.fit_transform(y_train)
# Y_test = sc.fit_transform(y_test)


# #8 - model inşası (lineer regresyon)
# from sklearn.linear_model import LinearRegression
# lr=LinearRegression()
# lr.fit(X_train, Y_train)
# tahmin = lr.predict(X_test)



#multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred=regressor.predict(x_test)
print(y_pred)



# #non-standard tahmin
# lr.fit(x_train, y_train)
# tahmin2 = lr.predict(x_test)


# #veri görselleştirme
# #sıralama
# x_train = x_train.sort_index()
# y_train = y_train.sort_index()
# #çizdirme
# plt.plot(x_train, y_train)
# plt.plot(x_test, lr.predict(x_test))
# #etiketleme
# plt.title("aylara göre satış")
# plt.xlabel("aylar")
# plt.ylabel("satışlar")


#backward eleminiation
import statsmodels.api as sm
#verinin en başına 1 bias kolonu eklenir. y=ALFA_BIAS+bx..
X = np.append(arr=np.ones((14, 1)).astype(int), values = sonveriler.iloc[:, :-1], axis = 1)
#ilk başta veriden full 6 kolonu al
X_l = sonveriler.iloc[:, [0,1,2,3,4,5]]
X_l = np.array(X_l, dtype=float)  #floata dönüştür
#istatistikmodelden hangi X_l değişkenlerinin boy sonucunu ne kadar etkilediğini bul
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit() #♣model raporunu verir
print(model.summary()) #yazdırılan P değerlerinden en yüksek P yi eleriz



#1. kolonu atıp be yi aynen tekrarla
sonveriler = sonveriler.iloc[:, 1:]
X_l = sonveriler.iloc[:, [0,1,2,3,4]]
X_l = np.array(X_l, dtype=float)  #floata dönüştür
#istatistikmodelden hangi X_l değişkenlerinin boy sonucunu ne kadar etkilediğini bul
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit() #♣model raporunu verir
print(model.summary()) #yazdırılan P değerlerinden en yüksek P yi eleriz



#p value'ler yüksek olduğundan kalan verilerle yeni bir eğitme yapalım.
#windy kolonunu atalım
x_train = x_train.iloc[:, 1:]
x_test = x_test.iloc[:, 1:]
regressor.fit(x_train, y_train)
y_pred=regressor.predict(x_test)
