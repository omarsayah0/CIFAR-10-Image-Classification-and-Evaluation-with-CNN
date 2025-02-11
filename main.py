#This is how to run the program : python -m streamlit run main.py

import numpy as np
import streamlit as st
from utils import (
    load_data, 
    load_model, 
    show_classification_report, 
    show_conf_matrix, 
    show_roc, 
    show_samples, 
    predict_uploaded_image
)

st.set_page_config(page_title="CIFAR-10 Classifier", layout='centered')

def main():
    st.title("CIFAR-10 Image Classification & Evaluation")
    model = load_model()
    tab1, tab2 = st.tabs(["📊 Model Evaluation", "🖼️ Predict Uploaded Image"])
    with tab1:
        st.subheader("Model Evaluation on CIFAR-10 Test Data")
        x_test, y_test = load_data()
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        st.write(f"❌ Test Loss: {loss:.2f}")
        st.write(f"✅ Test Accuracy: {acc:.2f}")
        y_pred_proba = model.predict(x_test)
        y_pred = np.argmax(y_pred_proba, axis = 1)
        st.subheader("Sample images with predictions")
        show_samples(x_test, y_test, y_pred)
        st.subheader("Classification Report")
        show_classification_report(y_test, y_pred)
        st.subheader("Confution Matrix")
        show_conf_matrix(y_test, y_pred)
        st.subheader("Roc Curve")
        show_roc(y_test, y_pred_proba)
    with tab2:
        st.subheader("Upload an Image for Prediction")
        predict_uploaded_image(model)

if __name__ == '__main__':
    main()