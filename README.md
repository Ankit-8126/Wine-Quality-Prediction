# 🍷 Wine Quality Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a wine is **Good Quality** or **Bad Quality** using Machine Learning. The model is trained on the Wine Quality dataset and deployed as an interactive web application using **Streamlit**.

The project demonstrates the complete Machine Learning workflow, including data preprocessing, exploratory data analysis (EDA), model training, model comparison, hyperparameter tuning, feature importance analysis, and deployment.

---

## 🎯 Problem Statement

The objective of this project is to classify wines into:

- ✅ Good Quality (Quality ≥ 7)
- ❌ Bad Quality (Quality < 7)

using their physicochemical properties.

---

## 📂 Dataset

**Dataset:** Wine Quality Dataset

Features used:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

Target Variable:

- quality_label
  - 1 = Good Quality
  - 0 = Bad Quality

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Git
- GitHub

---

## 📊 Machine Learning Workflow

### 1. Data Loading

- Loaded the Wine Quality dataset.

### 2. Exploratory Data Analysis (EDA)

- Displayed first five rows
- Dataset information
- Statistical summary
- Missing value analysis

### 3. Correlation Analysis

- Generated a correlation matrix.
- Visualized relationships using a heatmap.

### 4. Data Preprocessing

- Created binary target:
  - Good Quality = 1
  - Bad Quality = 0

### 5. Train-Test Split

- 80% Training
- 20% Testing

### 6. Model Training

Models trained:

- Logistic Regression (Without Scaling)
- Logistic Regression (With StandardScaler)
- K-Nearest Neighbors (KNN)
- Decision Tree

### 7. Hyperparameter Tuning

Optimized the KNN model using **GridSearchCV**.

### 8. Feature Importance

Used a Decision Tree model to visualize the most important features affecting wine quality.

### 9. Deployment

Built and deployed an interactive web application using Streamlit.

---

# 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression (Without Scaling) | *Your Accuracy* |
| Logistic Regression (With Scaling) | *Your Accuracy* |
| Decision Tree | *Your Accuracy* |
| **Optimized KNN** | **92.19%** |

---

# 📊 Final Evaluation (Best KNN Model)

Accuracy

```
92.19%
```

Precision

```
78.95%
```

Recall

```
63.83%
```

F1 Score

```
70.59%
```

Confusion Matrix

```
[[265   8]
 [ 17  30]]
```
---

# 📁 Project Structure

```
Wine-Quality-Prediction/
│
├── app.py
├── train_model.py
├── winequality.csv
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
      ├── home.png
      ├── prediction.png
      ├── heatmap.png
      └── feature_importance.png
```

---

# ▶️ How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/mrsonu-boop/Wine-Quality-Prediction.git
```

### Navigate to the Project Folder

```bash
cd Wine-Quality-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

---

# 🚀 Live Demo

**Streamlit App:**

https://wine-quality-predictions.streamlit.app/
```

---

# 💻 GitHub Repository

```
https://github.com/mrsonu-boop/Wine-Quality-Prediction
```

---

# 🔮 Future Improvements

- Try XGBoost and Random Forest.
- Improve recall for Good Quality wines.
- Add interactive visualizations.
- Deploy using Docker.
- Connect to a cloud database.

---

# 👨‍💻 Author

**Prathmesh Thorwat**

LinkedIn:

https://www.linkedin.com/in/prathmesh-thorwat-2699883b2?utm_source=share_via&utm_content=profile&utm_medium=member_android

GitHub:

https://github.com/mrsonu-boop

---

# ⭐ If you found this project useful

Please ⭐ Star this repository on GitHub.