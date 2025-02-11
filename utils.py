from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import seaborn as sns
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import streamlit as st 

@st.cache_data
def load_data():
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0
    y_test = y_test.reshape(-1)
    return (x_test, y_test)

def show_classification_report(y_test, y_pred):
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck", "accuracy", "macro avg", "weighted avg"]
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    report = classification_report(y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report)
    report.columns = class_names 
    sns.heatmap(report.iloc[: -1, : -2], annot=True, fmt=".2f", cmap="Blues")
    ax1.set_title("Classification report")
    st.pyplot(fig1)
    plt.close(fig1)

def show_conf_matrix(y_test, y_pred):
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels = class_names)
    disp.plot(cmap='Blues', ax=ax2)
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)
    plt.close(fig2)

def show_roc(y_test, y_pred_proba):
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    y_roc = label_binarize(y_test, classes=[i for i in range(10)])
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_roc[:, i], y_pred_proba[:, i])
        ax3.plot(fpr, tpr, label=f"Class {class_names[i]} (AUC = {roc_auc_score(y_roc[:, i], y_pred_proba[:, i]):.2f})")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("Roc Curve")
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)

def show_samples(x_test, y_test, y_pred):
    fig, axes = plt.subplots(3, 4, figsize = (15, 10), num="Samples")
    axes = axes.flatten()
    for i in range (12):
        axes[i].imshow(x_test[i])
        class_names = ["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]
        is_correct = class_names[y_test[i]] == class_names[y_pred[i]]
        color = "green" if is_correct else "red"
        text = f"{'correct' if is_correct else 'wrong'}"
        axes[i].set_title(f"pred: {class_names[y_pred[i]]}, True: {class_names[y_test[i]]}\n{text}", color= color)
        axes[i].axis('off')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

@st.cache_resource
def load_model():
    model = keras.models.load_model('cifar_cnn.keras')
    return model

def predict_uploaded_image(model):
    class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    st.write("**Keep in your mind the model can only predict these types:**")
    st.markdown("""
    - ✈️ **Airplane**
    - 🚗 **Automobile**
    - 🐦 **Bird**
    - 🐱 **Cat**
    - 🦌 **Deer**
    - 🐶 **Dog**
    - 🐸 **Frog**
    - 🐴 **Horse**
    - 🚢 **Ship**
    - 🚚 **Truck**
    """)
    uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                real_image = image
                image = image.convert("RGB").resize((32, 32))
                img_array = np.array(image).astype("float32") / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                st.subheader(f"**This is a :** {class_names[predicted_class]}")
                st.image(real_image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error processing the image: {e}")