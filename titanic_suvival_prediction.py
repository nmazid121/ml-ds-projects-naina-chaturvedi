import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Titanic.csv
# https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# data frame segments the data by rows and columns 

print("This is the initial data head :")
print(df.head())


print("Some information")
print(df.info())

# lets find the average age of a person from the dataframe

mean_age = df['Age'].mean()

# fill in all the missing values of the people with our new mean age

df['Age'] = df['Age'].fillna(mean_age)

print('Updated dataframe of our csv', df['Age'])

print(mean_age)

df['is_child'] = False
df['ages_above_18'] = 0


# populate the is child column to see if they are a child or adult
df['is_child'] = df['Age'] < 18

gender_mapping = {'male' : 1, 'female' : 0}

df['Sex'] = df['Sex'].map(gender_mapping)


print(df['Sex'])
print(df['is_child'])
print(df.info())
df = df.drop(columns=['Cabin'])

# fill embarked since it's empty
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

# encode the Embarked column because it is all filled with letters but it needs to be numbers
# lets make a dictionary

encoding = {'C' : 0, 'S' : 1, 'Q' : 2}
# apply the encoding to the actual column itself

df['Embarked'] = df['Embarked'].map(encoding)

print(df.isna().sum())

print(df['Embarked'].info())

# set the features used for testing the model

features = ['Pclass', 'Sex', 'Age', 'Fare', 'is_child', 'Embarked']

# create a dataframe that only takes in the columns that we need from the original dataframe from the features we selected

X = df[features]
y = df['Survived']

print(X)
print(y)

# training the model itself

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("The accuracy of the model is : ", accuracy_score(y_test, y_pred))

with open("titanic_prediction.pkl", "wb") as f :
    pickle.dump(model, f)