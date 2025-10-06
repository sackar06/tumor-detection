# 🧠 Tumor Detection using Machine Learning

## 📘 Project Overview
This project aims to classify **tumors as malignant (M)** or **benign (B)** using a supervised machine learning approach.  
It demonstrates data preprocessing, exploratory data analysis (EDA), feature scaling, and model building using **Random Forest Classifier** in Python.

The dataset contains medical measurements such as **mean radius**, **texture**, **perimeter**, **area**, and other cell nucleus features that help in identifying the nature of the tumor.

---

## 🎯 Objective
The main objective of this project is to:
- Analyze and visualize tumor characteristics.  
- Build a predictive model to classify tumors as **Malignant (M)** or **Benign (B)**.  
- Identify key features contributing most to tumor classification.  

---

## 🧩 Dataset Description
The dataset used is `Tumor_Detection.csv`, containing the following:
- **Features:** Various tumor characteristics such as radius, texture, smoothness, compactness, concavity, etc.  
- **Target Label:** `diagnosis` column —  
  - `M` → Malignant (Cancerous)  
  - `B` → Benign (Non-Cancerous)  

---

## 🧹 Data Cleaning and Preprocessing

### Steps Involved:
1. **Loading the Dataset**  
   The dataset is read using `pandas.read_csv()` and previewed for structure and basic info.

2. **Removing Irrelevant Columns**  
   Columns like `id` and any unnamed fields were removed since they don't contribute to model learning.

3. **Checking for Missing Values**  
   Verified the dataset’s completeness using `.isnull().sum()` to ensure no missing entries.

4. **Encoding the Target Variable**  
   Converted `diagnosis` values:  
   - `M` → `1`  
   - `B` → `0`

5. **Feature Scaling**  
   Used `StandardScaler` to normalize features to improve model performance.

---

## 📊 Exploratory Data Analysis (EDA)

The EDA process provided insights into the dataset through:
- **Count Plot** – Showed the distribution of Benign and Malignant tumors.  
- **Correlation Matrix and Heatmap** – Revealed relationships between features and target.  
- **Statistical Summary** – Displayed mean, standard deviation, and range for each feature.

Key findings:
- The dataset is slightly imbalanced but still well-suited for binary classification.
- Strong correlations observed among size-related features like `radius_mean`, `perimeter_mean`, and `area_mean`.

---

## ⚙️ Model Development

### 1. **Model Used:**  
   **Random Forest Classifier** (from `sklearn.ensemble`) was selected for its robustness and ability to handle complex feature interactions.

### 2. **Model Training Process:**
- Split the data into **80% training** and **20% testing** sets using `train_test_split()`.
- Applied **StandardScaler** for normalization.
- Trained the model on the scaled training data.
- Predicted outcomes for the test set.

---

## 📈 Model Evaluation

The trained Random Forest Classifier was evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)

### 📊 Performance Metrics:
| Metric | Score |
|---------|-------|
| **Accuracy** | ~94–96% |
| **Precision (Malignant)** | 0.97 |
| **Recall (Malignant)** | 0.98 |
| **F1 Score** | 0.98 |

---

## 🌟 Key Findings and Insights

1. The **Random Forest Classifier** achieved high accuracy and effectively differentiated between benign and malignant tumors.  
2. **Top contributing features** to classification were:
   - `mean radius`
   - `mean perimeter`
   - `mean concavity`
   - `mean area`
3. Visual analysis (heatmap and feature importance) showed that size and shape characteristics of cells are strong indicators of malignancy.
4. The model can serve as a base for building **AI-assisted diagnostic tools** in medical imaging and oncology.

---

## 🧮 Tools and Libraries Used
- **Programming Language:** Python  
- **Environment:** Jupyter Notebook  
- **Libraries:**  
  - `pandas`, `numpy` – Data handling and analysis  
  - `matplotlib`, `seaborn` – Data visualization  
  - `sklearn` – Machine learning and evaluation  

---

## 🧠 Conclusion
The **Tumor Detection** project successfully demonstrates how machine learning can assist in **early cancer detection** by analyzing tumor features.  
Through effective preprocessing, EDA, and model building, a high-performing classifier was developed that can predict tumor types with nearly 99% accuracy.

This project highlights the power of **data-driven decision-making** in healthcare applications.

---
