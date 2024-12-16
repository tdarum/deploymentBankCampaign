import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

def load_model():
    model_path = "catboost_model2.pkl"
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def preprocess_data(data, categorical_features):
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].astype(str)  
    return data

# Display SHAP Summary Plot
def plot_shap_values(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.summary_plot(shap_values, data, show=False)
    st.pyplot(plt)

# Main Streamlit App
def main():
    # Load Model 
    model = load_model()
    categorical_features = ['job', 'housing', 'loan', 'contact', 'month', 'poutcome']

    # Load and Display Logo
    st.image("logo.jpg", use_column_width=True)
    st.title("Bank Marketing Campaign Prediction App")

    # Sidebar 
    st.sidebar.title("Choose Prediction Mode")
    prediction_choice = st.sidebar.radio(
        "Select an option:",
        ("Single Prediction", "Bulk Prediction")
    )

    if prediction_choice == "Single Prediction":
        # Single Prediction Section
        st.header("Single Prediction")

        user_input = {}
        user_input['age'] = st.number_input("Age", min_value=18, max_value=100, step=1)
        user_input['job'] = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                                                 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
        user_input['balance'] = st.number_input("Balance", step=1)
        user_input['housing'] = st.selectbox("Housing Loan", ['yes', 'no'])
        user_input['loan'] = st.selectbox("Personal Loan", ['yes', 'no'])
        user_input['contact'] = st.selectbox("Contact Communication Type", ['unknown', 'cellular', 'telephone'])
        user_input['month'] = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        user_input['campaign'] = st.number_input("Campaign (Number of Contacts Performed)", min_value=1, step=1)
        user_input['pdays'] = st.number_input("Pdays (Days Since Last Contact)", min_value=-1, step=1)
        user_input['poutcome'] = st.selectbox("Previous Campaign Outcome", ['unknown', 'success', 'failure', 'other'])

        # Predict button
        if st.button("Predict"):
            # Convert input to DataFrame
            input_data = pd.DataFrame([user_input])
            input_data = preprocess_data(input_data, categorical_features)

            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_label = "Yes" if prediction == 1 else "No"
            st.write(f"Prediction: {prediction_label}")

            # Option to retry
            if st.button("Try Again"):
                st.experimental_rerun()

    elif prediction_choice == "Bulk Prediction":
        # Bulk Prediction 
        st.header("Bulk Prediction")
        bulk_file = st.file_uploader("Upload CSV for Bulk Prediction", type=["csv"])

        if bulk_file:
            bulk_data = pd.read_csv(bulk_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(bulk_data.head())

            #  bulk data
            bulk_data = preprocess_data(bulk_data, categorical_features)

            #  predictions
            try:
                bulk_predictions = model.predict(bulk_data)
                bulk_data['Prediction'] = ['Yes' if pred == 1 else 'No' for pred in bulk_predictions]

                st.write("Deposit Prediction:")
                st.dataframe(bulk_data)

                csv = bulk_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="bulk_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")


if __name__ == "__main__":
    main()
