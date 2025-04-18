## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

Done by : ROHIT GP - 212224220082 / 24900185

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Encoding Data (2).csv")
df
```
![Screenshot 2025-04-18 082620](https://github.com/user-attachments/assets/23b5d6de-4b6d-4fda-aeac-6c38fe2d6502)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2025-04-18 082741](https://github.com/user-attachments/assets/b5792348-e1c4-4c10-94af-583abf6fc7b4)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2025-04-18 082832](https://github.com/user-attachments/assets/ca40ee8d-360e-41cd-aa06-e76ce1b43be5)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2025-04-18 082837](https://github.com/user-attachments/assets/61a8ed81-e253-4e0c-93a8-0d16897b85ca)
```
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2025-04-18 082845](https://github.com/user-attachments/assets/2a9780ef-31d2-484e-801f-a2b0e4b9e1eb)
```
pd.get_dummies(df2,columns=['nom_0'])
```
![Screenshot 2025-04-18 082851](https://github.com/user-attachments/assets/8965bd29-d87a-4c6e-9be3-8a10350f071e)
```
pip install --upgrade category_encoders
```
![Screenshot 2025-04-18 100201](https://github.com/user-attachments/assets/43e41bb3-ae5c-42ad-b7ef-057ef1e19170)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2025-04-18 082906](https://github.com/user-attachments/assets/7bee8afd-1a46-4df1-bcd3-677e4a7108eb)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2025-04-18 082912](https://github.com/user-attachments/assets/3e4d5936-3f17-4baa-849f-715ec77c2004)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2025-04-18 082919](https://github.com/user-attachments/assets/aad1232f-2a6f-4824-a62f-82cc2f8ca4f9)
```
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```
![Screenshot 2025-04-18 082924](https://github.com/user-attachments/assets/a31bf316-c978-4725-b062-62876cfbf83e)
```
df.skew()
```
![Screenshot 2025-04-18 082930](https://github.com/user-attachments/assets/0c51cc95-2b07-40ed-9ade-406789acd249)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 082937](https://github.com/user-attachments/assets/54634c93-4b54-468e-a243-b74fc0c4349a)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-18 082945](https://github.com/user-attachments/assets/d20e2ac4-3292-482b-8b95-5fbd61722875)
```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 082950](https://github.com/user-attachments/assets/0fb1c3d4-b080-4e03-b1c5-dae6b3d73cd9)
```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-18 082956](https://github.com/user-attachments/assets/96dddaed-e2b7-4f70-a11f-f4e07793c42b)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2025-04-18 083001](https://github.com/user-attachments/assets/8b8a6bdf-ff7c-4b68-bc12-3ca25463a2f6)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![Screenshot 2025-04-18 083007](https://github.com/user-attachments/assets/82baf864-f923-4ca9-9917-951c750a8124)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![Screenshot 2025-04-18 083014](https://github.com/user-attachments/assets/38b8b0d1-67d0-4387-a53b-c5cf4b7bed5a)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![Screenshot 2025-04-18 083020](https://github.com/user-attachments/assets/9206fb7b-3740-4bfc-91fc-07e4e80fdc4e)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![Screenshot 2025-04-18 083031](https://github.com/user-attachments/assets/682bb5f7-4218-4577-b955-6ead0d4865f8)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2025-04-18 083037](https://github.com/user-attachments/assets/183bad17-6997-4cb1-9d44-d331af7cfb90)
```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show() 
```
![Screenshot 2025-04-18 083043](https://github.com/user-attachments/assets/776158b7-4fdf-46fc-bcac-74d52b6b69ea)
```
df['Highly Negative Skew_1']=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![Screenshot 2025-04-18 083048](https://github.com/user-attachments/assets/ce9d4547-e95c-4dd0-8011-e46a83f35f2e)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![Screenshot 2025-04-18 083053](https://github.com/user-attachments/assets/6a73bdd4-53cd-4f98-8e42-4b05fa2b6b5e)
```
dt=pd.read_csv("/content/titanic_dataset (2).csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![Screenshot 2025-04-18 083058](https://github.com/user-attachments/assets/93b22804-0039-4453-a62c-4f439f6e0e86)

# RESULT:
Successfully read the given data and performed Feature Encoding and Transformation process.       
