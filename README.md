# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import numpy as np
from scipy import stats
import pandas as pd
df=pd.read_csv('/content/bmi.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/30ad9d30-db18-4cd9-b00c-84d834b04776)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/a6a51dff-02a4-4f4d-a6a8-c2475905d36b)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
```
![image](https://github.com/user-attachments/assets/7b04734d-6997-424d-abb6-07ca095282a2)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Head','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/1a307bc3-bf4e-4feb-aabe-fc5b25d8d2aa)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/078254e1-dc99-4686-adc7-7aba6b4b1247)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/c6a95201-86ab-4b14-9333-24e9ce63af62)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/2e71ecc5-7cc4-46c6-b08e-a7d28066f90d)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/5b823ed8-1b4f-4627-8897-ef92b774f0bd)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![image](https://github.com/user-attachments/assets/74c49f2a-0045-405a-9de8-cce1c2138edd)
```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/21f6f978-10d5-4669-bed4-e6138308a35d)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head
```
![image](https://github.com/user-attachments/assets/e73399b7-1fb3-414f-818f-8702a48d4fd3)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/3ffecfae-3de3-42d9-a54b-2a63798beeee)
```
chi2, p, _, _=chi2_contingency(contingency_table)
print("Chi-Square Statistic: {chi2}")
print(f"P-value:{p}")
```
![image](https://github.com/user-attachments/assets/63b2ae59-6a26-4933-b99b-724fb4d91923)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
