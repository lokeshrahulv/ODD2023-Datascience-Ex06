# EX-06 FEATURE TRANSFORMATION

## Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

## Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## Algorithm:
Step1: Read the given Data.
Step2: Clean the Data Set using Data Cleaning Process.
Step3: Apply Feature Transformation techniques to all the features of the data set.
Step4: Print the transformed features.

## Program:
```
Developed By: Lokesh Rahul V V
Register No: 212222100024
```
## Importing libraries and reading csv file:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
df=pd.read_csv("Data_to_Transform.csv")
```
## Basic Information
```python
df.head()
df.info()
df.info()
```
![274812523-e1d70826-3e9e-4413-a1c3-ef9c3dc8747d](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/0c9912fc-63ba-4c7b-86a0-9d3b18b65b65)![274812538-98818400-9ff3-4ad7-9ccf-ae9fa1129bb4](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/ed834ace-7647-4701-85cd-bd9197e3d874)
!['](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/e7f5d264-6705-4a20-ba5c-02dfb04d7c6e)
## Before Transformation:
```python
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![274877477-aa5c908b-6571-4164-8e97-9bd5d6a7d22d](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/75963c13-f367-4c21-a600-864987b46ab5)
![274877505-86450d9c-b6cc-4402-8818-4794452776f6](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/a5d5cdad-80c5-4859-bd31-c8845fa4fb05)
![274877544-71b58850-8cea-4ae7-af4e-cee23a3597c1](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/7af5ca2e-06ee-4801-9871-2ed94b23cc60)

## Log Transformation:
```python
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()
```
![274877622-40eb3b47-5fab-4bcc-b29b-166a14481d07](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/83801d96-b480-466b-85b4-99ed7e7901c9)
![274877628-3db41bdf-7563-4192-bbc7-298ba9836c9e](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/30ea5835-55d8-4e62-975b-04e3a8daa198)

## Reciprocal Transformation:
```python
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![274877681-ff507c37-5cdf-49f5-bdd8-26ad510444fb](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/f622214e-71b3-4f28-a047-1dfabbb3ad29)

## SquareRoot Transformation:
```python
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()
```
![274877715-4bbde658-4fb6-44c2-a810-256118b7986a](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/1f15abf4-7424-4bea-b18f-8ae27cd6bc98)
## Power Transformation:
```python
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()
```
![274877750-af6ffcb2-0549-41d4-a394-0447731adea0](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/d43f8d46-afbc-4ab1-a42f-12f6f46eee2e)
![274877758-2a2955c9-5c2d-425d-b87a-1ed942cc772b](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/f74b0f38-eed0-4186-acfb-e5cbb01b685d)

## Quantile Transformation:
```python
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.title("Moderate  Negative Skew")
plt.show()'
```
![274877815-9fd9482d-f704-453c-a0b7-1aa52561e39f](https://github.com/lokeshrahulv/ODD2023-Datascience-Ex06/assets/118423842/13655062-5410-4ab5-bbda-d574d4741b17)
## Result:
Thus feature transformation is done for the given dataset.
