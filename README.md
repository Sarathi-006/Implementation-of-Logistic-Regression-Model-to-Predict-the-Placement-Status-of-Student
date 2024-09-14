# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S.PARTHASARATHI
RegisterNumber:  22223040144
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
PLACEMENT DATA :
![267753589-cba641d7-4b64-474a-9df3-f8047b4ddc21](https://github.com/user-attachments/assets/34587395-c59d-4e49-b179-9cb718b9793f)
SALARY DATA :
![Screenshot 2024-09-14 081235](https://github.com/user-attachments/assets/29c0b085-a480-4cf6-9be9-f2ea7e76acf4)
CHECKING THE NULL() FUNCTION :
![267753782-196a08f0-0571-40f2-bfdf-b6e1d2b4fa8f](https://github.com/user-attachments/assets/83a42d0c-06c3-4a3d-aa86-c8b999582504)
DATA DUPLICATE :
![267753891-3efb2a8c-6c60-4466-99b2-2c3c7b7a39b4](https://github.com/user-attachments/assets/f7447530-65d8-45cc-8c76-7aa390b19078)
PRINT DATA :
![267753963-37d05f23-2187-49d2-a871-7dbf5d7baca9](https://github.com/user-attachments/assets/8d335070-fb0c-4a19-804e-b53f16540c16)
DATA STATUS :
![267754049-d0b24ebb-4d7a-4956-b6e5-b87f65ccbeeb](https://github.com/user-attachments/assets/16cfb0ad-7c28-48b7-903b-a39d7ec9d9f2)
Y-PREDICTION ARRAY :
![267754328-81a5cd80-1fa0-48d8-a838-567b6e7a6676](https://github.com/user-attachments/assets/2995ebec-8b91-4252-9052-b6018a2ea9eb)
ACCURACY VALUE :
![267754448-1ca21819-8baa-4312-aae8-1b094fe75ea6](https://github.com/user-attachments/assets/558b751a-30b1-461d-b4b2-d4e50cf5e33f)
CONFUSION ARRAY :
![267754513-675efabe-006d-463a-b5f0-0cc4354ca37a](https://github.com/user-attachments/assets/acc994d0-5fee-4980-92b9-c5285b29ecb9)
CLASSIFICATION REPORT :
![267754597-be3ab929-d71c-492a-8adc-9a054cf08983](https://github.com/user-attachments/assets/9dd9e5e8-dbaf-4f77-88ca-652c032f46e7)
PREDICTION OF LR :
![267754663-295b82c5-385c-4832-9d92-282a651946cb](https://github.com/user-attachments/assets/4dfe234b-b5ad-42d4-b719-77ccbc2441c8)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
