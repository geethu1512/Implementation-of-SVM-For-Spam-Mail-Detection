<img width="202" height="81" alt="y shape" src="https://github.com/user-attachments/assets/de926b4b-cbfc-4713-bc75-151ad2f804fc" /><img width="790" height="277" alt="data" src="https://github.com/user-attachments/assets/ede89eb5-3b3c-4bd4-b232-7fd0aab4cd89" /># Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages. 
2.Analyse the data. 
3.Use modelselection and Countvectorizer to preditct the values. 
4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: GEETHU R
RegisterNumber:  212224040089
*/
```
~~~
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from IPython.display import display, Markdown

data = pd.read_csv(r"C:\Users\admin\Downloads\spam.csv", encoding='Windows-1252')
display(Markdown("## DATA:"))
display(data.head())

display(Markdown(f"## data.shape():\n```text\n{data.shape}\nimage\n```"))

x, y = data['v2'].values, data['v1'].values
display(Markdown(f"## x.shape():\n```text\n{x.shape}\nimage\n```"))
display(Markdown(f"## y.shape():\n```text\n{y.shape}\nimage\n```"))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
display(Markdown(f"## x_train:\n```text\nimage\n```"))
display(Markdown(f"## x_train.shape():\n```text\n{x_train.shape}\nimage\n```"))

cv = CountVectorizer()
x_train, x_test = cv.fit_transform(x_train), cv.transform(x_test)
svc = SVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
display(Markdown(f"## y_pred:\n```text\nimage\n```"))

acc = accuracy_score(y_test, y_pred)
display(Markdown(f"## acc (Accuracy):\n```text\n{acc}\nimage\n```"))

con = confusion_matrix(y_test, y_pred)
display(Markdown(f"## con (Confusion Matrix):\n```text\n{con}\nimage\n```"))

cl = classification_report(y_test, y_pred)
display(Markdown(f"## cl (Classification Report):\n```text\n{cl}\nimage\n```"))

~~~

## Output:

### DATA:
<img width="790" height="277" alt="data" src="https://github.com/user-attachments/assets/52a91e56-88d4-4329-bb3a-9ea07ff54b2b" />

### data.shape():
<img width="220" height="72" alt="data shape" src="https://github.com/user-attachments/assets/7c03d8df-1686-4c9b-a889-535264290c10" />

### y.shape():
<img width="202" height="81" alt="y shape" src="https://github.com/user-attachments/assets/08e4a663-dd86-4af1-9fda-917e15eb410e" />

### x_train:
<img width="487" height="207" alt="x train" src="https://github.com/user-attachments/assets/891cf98a-86e5-43fd-9d38-e9731e2ae39e" />

### x_train_shape:
<img width="243" height="78" alt="x train shapes" src="https://github.com/user-attachments/assets/91015908-fb00-4c50-9a05-9593199d25f1" />

### Accuracy:
<img width="242" height="40" alt="accuracy" src="https://github.com/user-attachments/assets/51484889-b12a-43c3-94f5-04dba1f601ba" />

### Confusion matrix:
<img width="142" height="62" alt="confusion matrix" src="https://github.com/user-attachments/assets/3fcd097e-d2d9-4bd8-bada-236a46ec25f7" />

### Classification Report:
<img width="595" height="226" alt="classification report" src="https://github.com/user-attachments/assets/9c2a8783-7119-427b-b38f-6a99d5353144" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
