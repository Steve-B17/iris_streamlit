import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Page Configuration
st.set_page_config(page_title="Streamlit App", page_icon=":shark:", layout="wide")

# Title of the app
st.title("Simple Prediction App")

#Load dataset

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Input widgets

st.sidebar.subheader("Input features")
sepal_length = st.sidebar.slider("Sepal length" , 4.3, 7.9 , 5.8)
sepal_width = st.sidebar.slider ( " Sepal width" , 2.0 , 4.4 , 3.1)
petal_length = st.sidebar.slider ( " Petal length" , 1.0 , 6.9 , 3.8)
petal_width = st.sidebar.slider ( " Petal width" , 0.1 , 2.5 , 1.2)

#Separate x and y

X= df.drop("species", axis = 1)
Y= df.species

#Data Splitting

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#Model Building

rf= RandomForestClassifier(max_depth=2, max_features= 4, n_estimators=200 ,random_state=42)
rf.fit(X_train, Y_train)

#Apply to make predictions

y_pred = rf.predict([[sepal_length, sepal_width, petal_length, petal_width]])

#Print EDA

st.subheader("Brief EDA")
st.write('The data is grouped by the class and the variable mean is computed for each class')
groupby = df.groupby('species').mean()
st.write(groupby)
st.line_chart(groupby.T)

#Print the input features

input_features = pd.DataFrame({"Sepal length": sepal_length, "Sepal width": sepal_width, "Petal length": petal_length, "Petal width": petal_width}, index=[0])
st.write("Input features")
st.write(input_features)

#Print the prediction

st.subheader("Prediction")
st.metric("Predicted class", y_pred[0], '')

