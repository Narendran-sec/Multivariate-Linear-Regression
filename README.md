# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas as pd
### Step2
read the csv file

### Step3
get the values of x and y variables

### Step4
create the linear regression model and fit

### Step5
print the predicted output.

## Program:
```
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data with two independent variables
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])

# Create and train the multivariate linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict output for new data
new_data = np.array([[4, 5]])
predicted_output = model.predict(new_data)

print(f'Predicted Output: {predicted_output}')
```
## Output:
![10th output](https://github.com/Narendran-sec/Multivariate-Linear-Regression/assets/147473131/4812af6e-4bc9-44d0-8bf2-ee61c4e5ea3d)


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
