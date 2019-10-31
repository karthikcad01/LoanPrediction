
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import seaborn as sns
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


Loan = pd.read_csv('traindata.csv')
Loan.shape

#Finding null values and data imputation
Loan.isnull().sum()
Loan = Loan.fillna({"Gender":"Male", "Self_Employed":"No","Married":"Yes"})
Loan.drop('Loan_ID',axis=1,inplace=True)
Loan.Dependents[Loan.Dependents=='3+'] = 3

from sklearn.preprocessing import LabelEncoder
var_mod = ["Gender","Married",'Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    Loan[i] = le.fit_transform(Loan[i])

#Calculating the percentage of Null values in the dataframe
Loan.isnull().sum()
Loan_null = pd.DataFrame((Loan.isnull().sum()),columns=['Null_Values'])
Loan_null['%ofNullValeues'] = ((Loan_null['Null_Values'])/614*100).sort_values(ascending=True)
Loan_null

#Replacing missing values of NAN values with 0 and mean

Loan['Dependents'].fillna(value=0,axis=0,inplace=True)
Loan['LoanAmount'].fillna(value=Loan['LoanAmount'].mean(),axis=0,inplace=True)
Loan['Loan_Amount_Term'].fillna(value=Loan['Loan_Amount_Term'].mean(),axis=0,inplace=True)
Loan['Credit_History'].fillna(value=Loan['Credit_History'].mean(),axis=0,inplace=True)
Loan.info()

Y_train = Loan['Loan_Status']
X_train = pd.concat([Loan['Gender'], Loan['Married'], Loan['Dependents'], Loan['Education'], Loan['Self_Employed'], Loan['ApplicantIncome'], Loan['CoapplicantIncome'], Loan['LoanAmount'], Loan['Loan_Amount_Term'], Loan['Credit_History'], Loan['Property_Area']],axis=1)
Y_train.shape, X_train.shape

models = []
seed=7 
scoring= 'accuracy'
models.append(('KNN', KNeighborsClassifier()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
    results.append(cv_results)
    names.append(name)
    msg= "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


Dataset = pd.read_csv('testdata.csv')
Dataset.shape

#Finding null values and data imputation
Dataset.isnull().sum()
Dataset = Dataset.fillna({"Gender":"Male", "Self_Employed":"No","Married":"Yes"})

Dataset.Dependents[Dataset.Dependents=='3+'] = 3

from sklearn.preprocessing import LabelEncoder
var_mod = ["Gender","Married",'Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    Dataset[i] = le.fit_transform(Dataset[i])

#Calculating the percentage of Null values in the dataframe
Dataset.isnull().sum()
Dataset_null = pd.DataFrame((Dataset.isnull().sum()),columns=['Null_Values'])
Dataset_null['%ofNullValeues'] = ((Dataset_null['Null_Values'])/614*100).sort_values(ascending=True)
Dataset_null

#Replacing missing values of NAN values with 0 and mean

Dataset['Dependents'].fillna(value=0,axis=0,inplace=True)
Dataset['LoanAmount'].fillna(value=Dataset['LoanAmount'].mean(),axis=0,inplace=True)
Dataset['Loan_Amount_Term'].fillna(value=Dataset['Loan_Amount_Term'].mean(),axis=0,inplace=True)
Dataset['Credit_History'].fillna(value=Dataset['Credit_History'].mean(),axis=0,inplace=True)
Dataset.info()

X_test = pd.concat([Dataset['Gender'], Dataset['Married'], Dataset['Dependents'], Dataset['Education'], Dataset['Self_Employed'], Dataset['ApplicantIncome'], Dataset['CoapplicantIncome'], Dataset['LoanAmount'], Dataset['Loan_Amount_Term'], Dataset['Credit_History'], Dataset['Property_Area']],axis=1)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_test = knn.predict(X_test)
array=Dataset.values
arr=np.array(array)
arr = arr.T
arr[11] = Y_test
arr = arr.T
print(arr)
l=list()
rows=367
for i in range(rows):
    l.append([arr[i][0], arr[i][11]])
print(l)
df = pd.DataFrame(l)
df.columns = ['Loan_id', 'Loan_status']
df.to_csv("sample_submission.csv", index=False )
