from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from setuptools.command.rotate import rotate
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
warnings.filterwarnings("ignore")
data = './data/verionislem.csv'
df = pd.read_csv(data)
df = df.drop(columns="ID")
print(df.describe().T)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe(include="object"))
# """ finding garbage values """
for i in df.select_dtypes(include="object"):
    print(i)
    print(df[i].value_counts())
    print("***"*10)

#histogram-boxplot
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(data=df, x=i)
    plt.show()
print(df.shape,df.info,df.describe().T)

# """Scatter Plot """
for i in ['Yaş', 'Şehir', 'Çalışma Saati', 'Memnuniyet', 'Departman',
       'Deneyim Yılı']:
    sns.scatterplot(data=df,x=i,y='Maaş')
    plt.xticks(rotation=45)
    plt.show()


#Correlation Matrix to visualize parameters
corr_matrix = df.select_dtypes(include="number").corr()

plt.figure(figsize = (12,10))
ax = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm",fmt=".2f")

plt.yticks(rotation=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

""" Null values """
imputer = KNNImputer()
df["Yaş"] = df["Yaş"].fillna(35)#df["Yaş"].fillna(df.["Yaş"].mean(), inplace = True)
df["Maaş"] = df["Maaş"].fillna(5500)#df["Maaş"].fillna(df.["Maaş"].median(), inplace=True)
# print(df["Yaş"],df["Maaş"],df["Şehir"])
# print(df.loc[:5, "Yaş":"Deneyim Yılı"])
# print(df.isnull().sum())
# print(df[['Yaş','Maaş']])
for i in df.select_dtypes(include="number").columns:
    df[i] = imputer.fit_transform(df[[i]])
    #df[['Yaş', 'Maaş']] = df[['Yaş', 'Maaş']].round(0).astype(int)
    if i in ['Yaş','Maaş']:
        df[i] = df[i].round(0).astype(int)
# print(df.isnull().sum())
#
# print(df[['Yaş','Maaş']])
# print(df.select_dtypes(include='number').columns)
"""Aykiri degerler quantile solution"""
def wisker(col):
    q1,q3 = np.percentile(col,[25,75])
    iqr = q3 - q1
    non_negative_columns = ['Deneyim Yılı','Yaş',"Çalışma Saati","Maaş"]
    lw = (q1 - (1.5 * iqr)).round(0).astype(int)
    uw = (q3 + (1.5 * iqr)).round(0).astype(int)
    for col in df.select_dtypes(include="number"):
        if col in non_negative_columns:
            lw = max(0,lw)
    # lw = max(0,lw)
    return lw,uw
print(wisker(df['Deneyim Yılı']))#it says -6 there is no such a thing like negative experience years so....
df['Deneyim Yılı'][0]=-1#testing
non_negative_cols = ['Deneyim Yılı','Yaş',"Çalışma Saati","Maaş"]

for i in df.select_dtypes(include="number"):
    lw, uw = wisker(df[i])
    # if i in non_negative_cols:
    #     lw = max(0,lw)
    df[i] = np.where(df[i] < lw, lw, df[i])  # Cap lower values
    df[i] = np.where(df[i] > uw, uw, df[i])  # Cap upper values

# print(df['Deneyim Yılı'][0])#it says -6 there is no such a thing like negative experience years so we have to keep lower value as zero(0)....
# print(wisker(df['Çalışma Saati']))
# Q1 = df["Çalışma Saati"].quantile(0.25)
# Q3 = df["Çalışma Saati"].quantile(0.75)
# IQR = Q3 - Q1
# alt_sinir = Q1 - (1.5 * IQR)
# ust_sinir = Q3 + (1.5 * IQR)

# df = df[(df["Çalışma Saati"] >= alt_sinir) & (df["Çalışma Saati"] <= ust_sinir)]

#Label encoding
# dummy = pd.get_dummies(data=df,columns=["Şehir","Departman"],drop_first=True,dtype=int)
# # dummy.asype(int)
# print(dummy.columns.value_counts().sum())
# corr_matrix = dummy.select_dtypes(include="number").corr()
# plt.figure(figsize=(20,18))
# ax = sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".1f",annot_kws={"size": 4.8},cbar_kws={"shrink":0.8})
# plt.xticks(rotation=15, ha='right')
# plt.yticks(rotation=0)
# plt.title("Correlation Matrix",pad=10)
# plt.tight_layout()
# plt.show()
label_encoder = LabelEncoder()
df["Şehir"] = label_encoder.fit_transform(df['Şehir'])
df["Departman"] = label_encoder.fit_transform(df["Departman"])

# #Scaling
scaler = MinMaxScaler()

df[["Maaş","Çalışma Saati","Deneyim Yılı"]] = scaler.fit_transform(df[["Maaş","Çalışma Saati","Deneyim Yılı"]])

