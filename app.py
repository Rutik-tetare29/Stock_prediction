
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stock Trend Predictor", layout="wide")

st.title("📈 Stock Market Trend Predictor using KNN")
st.markdown("Upload your stock CSV file to analyze and predict future trends based on past performance.")

# Upload CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(data.head())

    st.write("🧹 Missing Values Check")
    st.write(data.isnull().sum())

    data.dropna(inplace=True)

    st.subheader("📉 Closing Price Plot")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(data['Close'], label='Closing Price')
    ax.set_title("Stock Closing Price Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Feature engineering
    data['Open - Close'] = data['Open'] - data['Close']
    data['High - Low'] = data['High'] - data['Low']

    # Drop NA if any after feature creation
    data.dropna(inplace=True)

    # Features and labels
    X = data[['Open - Close', 'High - Low']]
    Y = np.where(data['Close'].shift(-1) > data['Close'], 1, -1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=44)

    # GridSearchCV to find best K
    st.subheader("🔧 Model Training with GridSearchCV")
    params = {'n_neighbors': list(range(1, 21))}
    model = KNeighborsClassifier()
    clf = GridSearchCV(model, params, cv=5)
    clf.fit(X_train, y_train)

    best_k = clf.best_params_['n_neighbors']
    st.success(f"Best number of neighbors found: {best_k}")

    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = final_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("📊 Predictions Sample")
    result_df = pd.DataFrame({'Actual': y_test[:10], 'Predicted': y_pred[:10]})
    st.dataframe(result_df)

else:
    st.warning("Please upload a CSV file to continue.")
