 
# Rainfall in India 1901-2015 Analysis

This project involves the analysis of rainfall data in India from 1901 to 2015. It includes data preprocessing, exploratory data analysis, and the development of a Linear Regression model to predict annual rainfall.

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Introduction

This project aims to analyze historical rainfall data in India to understand rainfall patterns and predict annual rainfall using a Linear Regression model.

## Importing Libraries

We start by importing the necessary Python libraries for data analysis and model development.

 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
 

## Importing Dataset

We load the rainfall data for India from a CSV file to begin our analysis.

 
data = pd.read_csv('/content/rainfall_in_india_1901-2015.csv')
data.head()
 

## Exploratory Data Analysis (EDA)

We explore the dataset to understand its structure and characteristics.

 
data.info()
 

This code snippet provides information about the dataset, including the number of entries, columns, data types, and non-null counts.

## Feature Engineering

We perform feature engineering, which includes data cleaning and handling of missing values.
 
data.dropna(inplace=True)
 

This code snippet removes rows with missing values in the dataset to ensure data quality.

## Model Development and Evaluation

We build and evaluate a Linear Regression model to predict annual rainfall.

 
telangana = data.loc[data['SUBDIVISION'] == 'TELANGANA']
x = telangana['YEAR']
x.drop(columns=['YEAR'])
y = telangana['ANNUAL']

from sklearn.linear_model import LinearRegression
model = LinearRegression()
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
model.fit(x, y)
 
This code section loads data for the Telangana subdivision, prepares the input features (`x`), and the target variable (`y`). It then fits a Linear Regression model to predict annual rainfall.
 
b = model.intercept_
m = model.coef_
plt.scatter(x, y)
plt.scatter(x, m*x+b)
plt.plot()
 

This code snippet calculates the intercept and coefficient for the Linear Regression model and plots a scatter plot of the data points with the regression line.

## Conclusion

In conclusion, this project involved the analysis of historical rainfall data in India. We explored data, performed feature engineering, and developed a Linear Regression model to predict annual rainfall. This model can be used to make predictions and understand rainfall patterns in the Telangana subdivision.

 
