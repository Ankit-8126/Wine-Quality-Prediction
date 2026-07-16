# 🍷 Wine Quality Prediction using Machine Learning

An end-to-end Machine Learning project that predicts whether a wine is of **Good Quality (1)** or **Bad Quality (0)** based on its chemical composition. This project features a clean, interactive user interface built with **Streamlit**.

---

## 👤 Developer Information
* **Name:** Ankit
* **GitHub Profile:** [@Ankit-8126](https://github.com/Ankit-8126)
* **Project Repository:** [Wine-Quality-Prediction](https://github.com/Ankit-8126/Wine-Quality-Prediction)

---

## 🚀 Features
* **Exploratory Data Analysis (EDA):** Insights on dataset structure, missing values, and correlation analysis using Seaborn heatmaps.
* **Feature Scaling:** Implemented `StandardScaler` to normalize features for distance-based models.
* **Model Comparison:** Trained and evaluated multiple classification models:
  * Logistic Regression (With & Without Scaling)
  * Decision Tree (Used also for Feature Importance)
  * K-Nearest Neighbors (KNN)
* **Hyperparameter Tuning:** Fine-tuned the KNN model using `GridSearchCV` to achieve optimal performance.
* **Model Deployment:** Saved the best-performing model and scaler using `joblib` and served them via an interactive **Streamlit** web application.

---

## 🛠️ Tech Stack
* **Language:** Python 3.14.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, Joblib
* **Web Framework:** Streamlit
* **Version Control:** Git & GitHub

---

## 📊 Dataset & Target Variable
The model uses the `winequality.csv` dataset. The target variable is engineered as a binary classification:
* **Good Quality (1):** Wine Quality rating $\ge$ 7
* **Bad Quality (0):** Wine Quality rating < 7

### Key Features Used for Prediction:
1. Fixed Acidity / Volatile Acidity / Citric Acid
2. Residual Sugar / Chlorides
3. Free Sulfur Dioxide / Total Sulfur Dioxide
4. Density / pH / Sulphates
5. Alcohol (Highest impact feature)

---

## 💻 Installation & Setup

Follow these steps to run the project locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/Ankit-8126/Wine-Quality-Prediction.git](https://github.com/Ankit-8126/Wine-Quality-Prediction.git)
cd Wine-Quality-Prediction
