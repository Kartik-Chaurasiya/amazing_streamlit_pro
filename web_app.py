import streamlit as st
import pandas as pd
import pickle

# show data or head of data
def show_data(dataset):
    data = pd.read_csv(f'data/{dataset}')
    st.text("Data")
    df_head = st.slider('No of Rows', 0, 100)
    st.dataframe(data.head(df_head))
    return data

def load_model(df_data):
    model = pickle.load(open(f'saved_model/{df_data}.sav', 'rb'))
    return model

def show_pred(df_data, data, model):
    if df_data == 'advertising':
        n1 = st.number_input('TV')
        n2 = st.number_input('Radio')
        n3 = st.number_input('Newspaper')
        prediction = model.predict([[n1,n2,n3]])
        st.subheader(f'Value Predicted: {prediction[0]}')
    if df_data == 'iris':
        n1 = st.number_input('sepal_length')
        n2 = st.number_input('sepal_width')
        n3 = st.number_input('petal_length')
        n4 = st.number_input('petal_width')
        prediction = model.predict([[n1,n2,n3,n4]])
        st.subheader(f'Flower: {prediction[0]}')
    if df_data == 'social':
        n1 = st.number_input('Age', 1, 100)
        n2 = st.number_input('EstimatedSalary', 1, data['EstimatedSalary'].max())
        prediction = model.predict([[n1,n2]])
        st.subheader(f'Purchased: {"Yes" if int(prediction[0]) == 1 else "No" }')
    if df_data == 'waiter_tip':
        n1 = st.number_input('total_bill')
        sex = st.selectbox('Sex', ["Male", "Female"])
        n2 = 1 if sex == 'Male' else 2
        smoker = st.selectbox('Smoker', ["Yes", "No"])
        n3 = 1 if smoker == 'Yes' else 0
        day = st.selectbox('Day', ["Thur", "Fri", "Sat", "Sun"])
        if day == 'Thur':
            n4 = 0
        elif day == 'Fri':
            n4 = 1
        elif day == 'Sat':
            n4 = 2
        else:
            n4=3
        time = st.selectbox('Time', ["Lunch", "Dinner"])
        n5 = 1 if time == 'Dinner' else 0
        n6 = st.number_input('size', 1, 4)
        prediction = model.predict([[n1,n2, n3,n4,n5,n6]])
        st.subheader(f'Tip: {prediction[0]}')

st.header('Data Science')
df_data = st.sidebar.selectbox('Pick Dataset', ["Future Sales Prediction", "Iris Flower Classification", "Social Media Ads Classification", "Waiter Tips Prediction"])
if (df_data) == "Future Sales Prediction":
    dataset = "advertising.csv"
    data = show_data(dataset)
    model = load_model(dataset.split('.')[0])
    show_pred(dataset.split('.')[0], data, model)
elif (df_data) == "Iris Flower Classification":
    dataset = "iris.csv"
    data = show_data(dataset)
    model = load_model(dataset.split('.')[0])
    show_pred(dataset.split('.')[0], data, model)
elif (df_data) == "Social Media Ads Classification":
    dataset = "social.csv"
    data = show_data(dataset)
    model = load_model(dataset.split('.')[0])
    show_pred(dataset.split('.')[0], data, model)
elif (df_data) == "Waiter Tips Prediction":
    dataset = "waiter_tip.csv"
    data = show_data(dataset)
    model = load_model(dataset.split('.')[0])
    show_pred(dataset.split('.')[0], data, model)


    
