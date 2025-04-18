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
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
```
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```
```
pd.get_dummies(df2,columns=['nom_0'])
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data (2).csv")
df
```
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
```
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```
```
df.skew()
```
```
np.log(df["Highly Positive Skew"])
```
```
np.reciprocal(df["Moderate Positive Skew"])
```
```
np.sqrt(df["Highly Positive Skew"])
```
```
np.square(df["Highly Positive Skew"])
```
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show() 
```
```
df['Highly Negative Skew_1']=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
```
dt=pd.read_csv("/content/titanic_dataset (2).csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```

# RESULT:
       # INCLUDE YOUR RESULT HERE

       
