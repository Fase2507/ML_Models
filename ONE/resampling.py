import pandas as pd
from sklearn.datasets import make_classification
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


warnings.filterwarnings('ignore')

df = pd.read_csv('./data/creditcard.csv')
# print(df['Class'].value_counts())

# plt.figure(figsize=(6,4))
# sns.countplot(x='Class',data=df)
# plt.title("class disstrubition")
# plt.show()

X = df.drop(["Class","Time"],axis=1)
y = df["Class"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_train,y_train)
#
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test,y_pred)
# f1 = f1_score(y_test,y_pred)
# print(y.value_counts())
# print(f"Accuracy: %{(acc * 100):.2f} \n"
#       f"F1: %{(f1*100):.2f}")

# cm = confusion_matrix(y_test,y_pred)
# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True,fmt='d', cmap="Blues",xticklabels=["non-fraud","fraud"],yticklabels=["non-fraud","fraud"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.show()
"""RAndom OVER"""
# ros = RandomOverSampler()
# X_res, y_res = ros.fit_resample(X_train,y_train)
#
# class_counts = y_res.value_counts()
# # print(class_counts)
#
# plt.figure(figsize=(8,6))
# plt.bar(class_counts.index, class_counts.values, color = ["blue",'orange'])
# plt.xticks([0,1],['Class 0','Class 1'])
# plt.ylabel('Class')
# plt.xlabel(" Count")
# plt.title("Class Distrubition after Resampling")
# plt.show()
# for i in range(len(y)):
#     if y[i] ==1:
#         print(i,y[i])
# model.fit(X_res, y_res)
# y_pred_ros = model.predict(X_test)
# acc_ros = accuracy_score(y_test,y_pred_ros)
# f1_ros = f1_score(y_test, y_pred_ros)
# print(f"Accuracy: %{(acc_ros * 100):.2f} \n",
#       f"F1: %{(f1_ros * 100):.2f}")
#
# cm = confusion_matrix(y_test,y_pred_ros)
# plt.figure(figsize=(6,4))
# sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",xticklabels=["non-fraud","fraud"],yticklabels=["non-fraud","fraud"])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title("Confusion matrix")
# plt.show()

"""SMOTE"""
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_train,y_train)
# class_count = y_resampled.value_counts()
#
#
# model.fit(X_resampled, y_resampled)
# y_pred_smote = model.predict(X_test)
# acc_score = accuracy_score(y_test,y_pred_smote)
# f1_scor = f1_score(y_test, y_pred_smote)
# print(f"Accuracy: %{(acc_score * 100):.2f} \n",
#       f"F1: %{(f1_scor * 100):.2f}")
#
# cm = confusion_matrix(y_test, y_pred_smote)
# plt.figure(figsize=(6,8))
# sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",xticklabels=['Non-fraud','Fraud'],yticklabels=['Non-fraud','Fraud'])
# plt.ylabel("Actual_labels")
# plt.xlabel("Predicted_labels")
# plt.title("Confusion Matrix")
# plt.show()


"""ADASYN"""
# adasyn = ADASYN(random_state=42)
# scaler= StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_resampled,y_resampled = adasyn.fit_resample(X_train_scaled, y_train)
# Class_counts = y_resampled.value_counts()
#
# model.fit(X_resampled,y_resampled)
#
# X_test_scaled = scaler.transform(X_test)
# y_pred_adasyn = model.predict(X_test_scaled)
# acc_adasyn = accuracy_score(y_test,y_pred_adasyn)
# f1_score_adasyn = f1_score(y_test, y_pred_adasyn)
# print(f"Accuracy: %{(acc_adasyn * 100):.2f} \n",
#       f"F1: %{(f1_score_adasyn * 100):.2f}")
#
# cm = confusion_matrix(y_test, y_pred_adasyn)
# plt.figure(figsize=(6,8))
# sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",xticklabels=["Non-fraud","Fraud"],yticklabels=['Non_fraud',"Fraud"])
# plt.ylabel("Predicted")
# plt.xlabel("Actual")
# plt.title("CMatrix")
# plt.show()

""" Random UNder  """

# under_sampler = RandomUnderSampler(random_state=42)
# X_resampled_under, y_resampled_under = under_sampler.fit_resample(X_train,y_train)
# clas_coount = y_resampled_under.value_counts()
# print(clas_coount)#394 394
# model.fit(X_resampled_under,y_resampled_under)
# y_pred_under = model.predict(X_test)
# acc_score_under = accuracy_score(y_test,y_pred_under)
# f1_score_under = f1_score(y_test, y_pred_under)
# print(f"Accuracy: %{(acc_score_under * 100):.2f} \n",
#       f"F1: %{(f1_score_under * 100):.2f}")
#
# cm = confusion_matrix(y_test, y_pred_under)
# plt.figure(figsize=(6,8))
# sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",xticklabels=["NON_fraud","Fraud"],yticklabels=["NON_fraud","Fraud"])
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Confusion Matrix")
# plt.show()

"""EditedNearestNeighbour """
# editedNN = EditedNearestNeighbours()
# X_res_NN, y_res_NN = editedNN.fit_resample(X_train,y_train)
# print(y_res_NN.value_counts())
# model.fit(X_res_NN,y_res_NN)
#
# y_pred_nn = model.predict(X_test)
# acc_score_nn = accuracy_score(y_test,y_pred_nn)
# f1_score_nn = f1_score(y_test, y_pred_nn)
# print(f"Accuracy: %{(acc_score_nn * 100):.2f} \n",
#       f"F1: %{(f1_score_nn * 100):.2f}")
#
# cm = confusion_matrix(y_test, y_pred_nn)
# plt.figure(figsize=(6,8))
# sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=["NonFraud","Fraud"],yticklabels=["NonFraud","Fraud"])
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Confusion Matrix")
# plt.show()

#COST SENSITIVE
model2 = LogisticRegression(max_iter=1000, random_state=42,class_weight="balanced")
# model2.fit(X_train,y_train)
# y_pred_ = model2.predict(X_test)
# acc_score_ = accuracy_score(y_test,y_pred_)
# f1_score_ = f1_score(y_test, y_pred_)
# print(f"Accuracy: %{(acc_score_ * 100):.2f} \n",
#       f"F1: %{(f1_score_ * 100):.2f}")
#
# cm = confusion_matrix(y_test, y_pred_)
# plt.figure(figsize=(6,8))
# sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=["NonFraud","Fraud"],yticklabels=["NonFraud","Fraud"])
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Confusion Matrix")
# plt.show()

class_weight={0:1,1:10}
model3 = LogisticRegression(max_iter=1000,random_state=42,class_weight=class_weight)
model3.fit(X_train,y_train)
y_pred_ = model3.predict(X_test)
acc_score_ = accuracy_score(y_test,y_pred_)
f1_score_ = f1_score(y_test, y_pred_)
print(f"Accuracy: %{(acc_score_ * 100):.2f} \n",
      f"F1: %{(f1_score_ * 100):.2f}")

cm = confusion_matrix(y_test, y_pred_)
plt.figure(figsize=(6,8))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=["NonFraud","Fraud"],yticklabels=["NonFraud","Fraud"])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show()