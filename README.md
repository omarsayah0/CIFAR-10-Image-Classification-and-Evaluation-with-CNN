# CIFAR-10 Image Classification using CNN

## About
This project builds and trains a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 object categories such as airplanes, cars, animals, and ships.
The model is trained using data augmentation, Batch Normalization, and Dropout to improve generalization and accuracy.
A full Streamlit web application is included.
The goal of this project is to provide a complete pipeline from model training → evaluation → interactive deployment, making it a practical and educational example for beginners and intermediate learners in deep learning and computer vision.

---

## Files
- `model.py` → Builds, trains, and saves the CNN model using the CIFAR-10 dataset.
- `utils.py` → Contains helper functions for evaluation, visualization, and predicting uploaded images.
- `main.py` → Runs the Streamlit web app for model evaluation and image prediction.
- `cifar_cnn.keras` → The saved trained CNN model used by the Streamlit application.

---

## Steps Included

### 1️⃣ Data Preprocessing
- Loading the CIFAR-10 Dataset

        The dataset is loaded directly from `keras.datasets.cifar10`.  
        It provides:
          - 50,000 training images
          - 10,000 test images  

- Converting Image Data to float32

        The images are originally stored as unsigned 8-bit integers (0–255).
  
        We convert them to `float32` to make them suitable for neural network computations

- Normalizing Pixel Values
  
        To help the model train faster and converge more stably, pixel values are normalized to the range [0, 1]

- Normalize pixel values to range [0, 1]

        Each pixel value is divided by 255.0. Normalization helps the model train faster and more stably by keeping all input values within a small range, preventing large gradients and improving convergence.

- Data Augmentation (Training Only)

        To improve generalization and reduce overfitting, we apply data augmentation using ImageDataGenerator.
        The augmentations include:

                 Rotation: up to 15 degrees

                 Width shifting: ±10%

                 Height shifting: ±10%

                 Horizontal flipping


---

###  2️⃣ Model Architecture (Simple Explanation)

| **Layer** | **Description** |
|-----------|----------------|
| **Conv2D (32 filters, 3×3, ReLU)** | Detects basic features like edges and color transitions. |
| **BatchNormalization** | Stabilizes and speeds up the learning process. |
| **Conv2D (32 filters, 3×3, ReLU)** | Learns more detailed low-level patterns. |
| **BatchNormalization** | Normalizes activations for smoother training. |
| **MaxPooling2D (2×2)** | Reduces spatial size while keeping important features. |
| **Dropout (0.25)** | Prevents overfitting by randomly dropping neurons. |
| **Conv2D (64 filters, 3×3, ReLU)** | Extracts more complex shapes and textures. |
| **BatchNormalization** | Keeps values stable during deeper learning. |
| **Conv2D (64 filters, 3×3, ReLU)** | Learns richer and more abstract patterns. |
| **BatchNormalization** | Improves convergence in deeper layers. |
| **MaxPooling2D (2×2)** | Reduces the feature map size again. |
| **Dropout (0.25)** | Helps the model generalize better. |
| **Conv2D (128 filters, 3×3, ReLU)** | Detects high-level features like object parts. |
| **BatchNormalization** | Ensures stable activation distributions. |
| **Conv2D (128 filters, 3×3, ReLU)** | Extracts very detailed and advanced visual patterns. |
| **BatchNormalization** | Helps the model train efficiently even with many filters. |
| **MaxPooling2D (2×2)** | Compresses features to reduce computation. |
| **Dropout (0.25)** | Reduces overfitting by dropping more neurons. |
| **Flatten** | Converts the feature maps into a single long vector. |
| **Dense (256, ReLU)** | Learns how extracted features combine to represent objects. |
| **BatchNormalization** | Keeps training stable for the dense block. |
| **Dropout (0.25)** | Improves model generalization. |
| **Dense (256, ReLU)** | Further learns high-level combinations of features. |
| **BatchNormalization** | Normalizes the layer outputs for better training. |
| **Dropout (0.25)** | Prevents overfitting in the final dense layers. |
| **Dense (10, Softmax)** | Produces the final probability for each of the 10 CIFAR-10 classes. |

**In short:**  
The CNN gradually learns visual patterns — starting from simple edges, then shapes and textures, and finally full objects like animals, vehicles, and more.



### 3️⃣ Model Training

- **Data Augmentation: ** → To help the model generalize better and avoid memorizing the training images, several transformations are applied on the fly:
        - 15° random rotations  
        - Horizontal flipping  
        - 10% width and height shifting
- These transformations create new variations of the images, allowing the model to learn more robust features.

- **Training Configuration: **
-         Optimizer: Adam  
          Loss Function: Sparse Categorical Crossentropy  
          Metrics: Accuracy  
          Batch Size: 64  
          Epochs: 50  

- **Regularization Techniques** → The model uses:
-         **BatchNormalization** to stabilize activations and speed up convergence  
          **Dropout (0.25)** after each block to reduce overfitting  
          **Three MaxPooling layers** to progressively reduce spatial dimensions


---


## How to Run

1- Install Dependencies:
  ```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn streamlit pillow

```

2-Run :

  ```bash
python -m streamlit run main.py
```

3- Application Preview :

<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_05_48_376" src="https://github.com/user-attachments/assets/70adb1cb-49d1-4689-8227-37b27bc84668" />
</p>

The interface shows a clean and modern Streamlit dashboard where the user can switch between two main sections:  
- **Model Evaluation**  
- **Predict Uploaded Image**

At the top, the page displays the project title *“CIFAR-10 Image Classification & Evaluation”* along with a tab bar for easy navigation.  
The layout uses a dark theme with centered content, making the evaluation results and visualizations easy to view and interact with.

---



<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_08_11_498" src="https://github.com/user-attachments/assets/cabc2352-6134-445a-a86b-f6bf2f7ec31b" />
</p>


This section displays a grid of test images along with their predicted and true labels.  
Correct predictions are shown in green, while incorrect ones appear in red.  
This visual preview helps you quickly see how well the model performs on real CIFAR-10 images and where it makes mistakes.

---




<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_08_24_422" src="https://github.com/user-attachments/assets/aa2d22d7-6dca-4ab2-8769-20bdbd80f77d" />
</p>


## Classification Report

This heatmap shows the precision, recall, and F1-score for each CIFAR-10 class.  
Darker colors indicate stronger performance, making it easy to see which classes the model recognizes well and which ones are more challenging.

---




<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_08_40_790" src="https://github.com/user-attachments/assets/f62a32b8-f50b-4374-9032-bad8fea08f08" />
</p>


## Confusion Matrix

The confusion matrix visualizes how often the model correctly predicts each class versus where it makes mistakes.  
Diagonal values represent correct predictions, while off-diagonal values reveal which classes are confused with each other.

---




<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_08_48_15" src="https://github.com/user-attachments/assets/d227b0ff-24ea-404f-b253-c8fee03c7092" />
</p>


## ROC Curve

The ROC Curve shows the model’s ability to distinguish each class across all decision thresholds.  
Each line represents a CIFAR-10 category, and the curves staying close to the top-left corner indicate strong performance.  
The AUC (Area Under the Curve) values displayed in the legend provide a numerical measure of how well the model separates each class, with higher values meaning better discrimination.

---




<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_08_52_385" src="https://github.com/user-attachments/assets/3a28baad-c8f6-445f-909f-ae48061e8ef7" />
</p>


## Upload Image for Prediction

In this section, users can upload any image (JPG, JPEG, PNG) and the model will classify it into one of the 10 CIFAR-10 categories.  
The page clearly lists all supported classes so users know what the model can predict.  
After uploading, the app processes the image, resizes it to 32×32, and displays both the original image and the predicted label.

---





<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_09_00_965" src="https://github.com/user-attachments/assets/392e2107-c6a5-4338-9742-942ffe124dc2" />
</p>


## Example Prediction

In this example, I tested the model using a photo of my own cat **Mezo** 😄.
The image is resized and processed by the model, and it correctly predicts the class as **Cat**.  
This shows that the model can handle real world images not just the low resolution CIFAR-10 samples.

---





<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_09_14_847" src="https://github.com/user-attachments/assets/cceb96f7-2ef9-4103-bb5e-1a9a1599b72c" />
</p>


## Another Example Prediction

Here, I tested the model using a photo of my pet bird **Kitaawa** 😄. 
After processing the image, the model correctly identifies it as a **Bird**.  
This example further shows that the classifier works well on real-life images, even when they come from outside the CIFAR-10 dataset.

---





<p align="center">
<img width="3801" height="2098" alt="Capture_2025_11_23_22_09_22_847" src="https://github.com/user-attachments/assets/fa0bd07d-b51d-4bc7-ad97-b4058aa85290" />
</p>


## Airplane Prediction Example

Here, I uploaded a clear photo of an airplane to test the model.  
The classifier correctly identifies it as an **Airplane**, showing that it can recognize objects with high confidence even when the image is much larger and sharper than the original CIFAR-10 samples.


---


 ## Author
  
  Omar Alethamat

  LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

  ## License

  This project is licensed under the MIT License — feel free to use, modify, and share with attribution.
