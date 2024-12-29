Name: VEDIKA SULE

Company: CODTECH IT SOLUTION

ID: CT08FGJ

Domain: DATA ANALYTICS

Duration: December 2024 to January 2025

Mentor: N. SANTHOSH

OVERVIEW OF PROJECT

PROJECT:PREDICTIVE MODELING WITH LINEAR REGRESSION



OBJECTIVE:

This project aims to analyze a retail sales dataset to:
Explore how customer characteristics like Age, Gender, and Product Category impact their spending.
Predict the Total Amount spent by customers using regression techniques.
Understand the relationships between features such as product quantity, pricing, and customer demographics.



KEY ACTIVITIES:

Data Preparation:

Load the dataset containing customer transactions, including details like Age, Gender, Product Category, Quantity, Price per Unit, and Total Amount.

Transform categorical variables like Gender and Product Category into numerical values using one-hot encoding.

Feature Selection:

Use attributes like Age, Gender, Product Category, Quantity, and Price per Unit as inputs (features).

Set Total Amount as the output (target variable) to be predicted.

Train-Test Data Split:

Divide the data into training (80%) and testing (20%) sets to build and validate the model.

Linear Regression Model:

Train a linear regression model using Age as the primary input feature.

Predict the Total Amount for the test data based on the model.

Model Evaluation:

Visualize predictions with a scatter plot and regression line to check how well the model fits the data.

Evaluate the model using metrics like Mean Squared Error (MSE) and R-squared (R²) for accuracy.



TECHNOLOGIES USED:

Python Libraries:

Pandas: For data manipulation and preparation.

NumPy: For numerical computations.

Matplotlib: For creating visualizations.

Scikit-learn: For splitting the data, building the linear regression model, and evaluating its performance.


Data Processing Techniques:

One-hot encoding for converting categorical variables into numerical format.

Linear regression for predicting customer spending.


Dataset:

The dataset includes details of customer transactions, such as demographics, product types, quantities, prices, and total amounts.

CODE

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

df=pd.read_csv('retail_sales_dataset.csv')

print(df)

x = data[['Age', 'Gender', 'Product Category', 'Quantity', 'Price per Unit']]  # Features

y = data['Total Amount']

x = pd.get_dummies(x, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(X_train, X_test, y_train, y_test)

X_train = X_train[['Age']]  # Select 'Age' as the feature

X_test = X_test[['Age']]

model= LinearRegression()

#Train the model on the training data

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Make predictions using the test data

y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='b', label='Actual Data')

plt.plot(X_test, y_pred, color='r', label='Regression Line')

plt.xlabel('Age')

plt.ylabel('Total Amount')

plt.legend()

plt.show()


from sklearn.metrics import mean_squared_error, r2_score

mse= mean_squared_error(y_test,y_pred)

r2= r2_score(y_test,y_pred)

print(f"Mean Squared Error:{mse:.2f}")

print(f"R_squared:{r2:.2f}")

![image](https://github.com/user-attachments/assets/53328aa5-8ae8-4a33-91b1-06649c8b54dd)


OUTPUT:

![Screenshot (142)](https://github.com/user-attachments/assets/88f5308c-33ef-44c3-be51-104de65f5384)
![Screenshot (143)](https://github.com/user-attachments/assets/91c3ee07-87c0-4e3f-a095-8a2157b387f3)
