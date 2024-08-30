<H3>ENTER YOUR NAME: Varsha.G</H3>
<H3>ENTER YOUR REGISTER NO. 212222230166</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd                                                 
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         
df.head()
df.isnull().sum()                                                   
df.duplicated().sum()                                              
df=df.drop(['Surname', 'Geography','Gender'], axis=1)               
scaler=StandardScaler()                                            
df=pd.DataFrame(scaler.fit_transform(df))
df.head()
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values                     
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)   
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)                     
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)                   
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/2a6b4772-9103-4591-b2ae-5dbd95fddb2a)

![image](https://github.com/user-attachments/assets/8233493d-cfe6-4b61-88e8-d0a1c7e9298e)

![image](https://github.com/user-attachments/assets/ddc02573-faf9-45c9-bccb-9c8fb05ba21a)

![image](https://github.com/user-attachments/assets/46c4776a-2d51-4dbe-a1a0-376fb928a6a2)

![image](https://github.com/user-attachments/assets/8ead25c0-436c-4cdf-b0d7-0fc820243f72)

![image](https://github.com/user-attachments/assets/7b5ebc34-3edd-43b8-8492-273e33d17db2)

![image](https://github.com/user-attachments/assets/7e179cdd-7b3d-4973-9b51-1d56fcb133e3)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


