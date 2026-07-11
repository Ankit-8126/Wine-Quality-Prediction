import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("winequality.csv")

# Create binary target
df["quality_label"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

# Features and target
X = df.drop(["quality", "quality_label"], axis=1)
y = df["quality_label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

import pandas as pd

# Load Dataset
df = pd.read_csv("winequality.csv")

# -----------------------------
# Basic Data Analysis (EDA)
# -----------------------------

print("="*50)
print("First 5 Rows")
print("="*50)
print(df.head())

print("\n")

print("="*50)
print("Dataset Information")
print("="*50)
print(df.info())

print("\n")

print("="*50)
print("Statistical Summary")
print("="*50)
print(df.describe())

print("\n")

print("="*50)
print("Missing Values")
print("="*50)
print(df.isnull().sum())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Correlation Analysis
# -----------------------------

print("\n")
print("=" * 50)
print("Correlation Matrix")
print("=" * 50)

correlation = df.corr()

print(correlation)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# -----------------------------
# Create Binary Target Variable
# -----------------------------

print("\n")
print("=" * 50)
print("Creating Binary Target (quality_label)")
print("=" * 50)

df["quality_label"] = df["quality"].apply(lambda x: 1 if x >= 7 else 0)

print(df[["quality", "quality_label"]].head(10))

print("\nDistribution of Target Variable:")
print(df["quality_label"].value_counts())

# -----------------------------
# Separate Features and Target
# -----------------------------

print("\n")
print("=" * 50)
print("Separating Features and Target")
print("=" * 50)

# Features (Independent Variables)
X = df.drop(["quality", "quality_label"], axis=1)

# Target (Dependent Variable)
y = df["quality_label"]

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Names:")
print(X.columns.tolist())


# -----------------------------
# Train-Test Split
# -----------------------------

print("\n")
print("=" * 50)
print("Train-Test Split")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features :", X_train.shape)
print("Testing Features  :", X_test.shape)
print("Training Labels   :", y_train.shape)
print("Testing Labels    :", y_test.shape)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Logistic Regression (Without Scaling)
# -----------------------------

print("\n")
print("=" * 50)
print("Logistic Regression (Without Scaling)")
print("=" * 50)

# Create model
lr_model = LogisticRegression(max_iter=1000)

# Train model
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression (Without Scaling)")

plt.show()

# -----------------------------
# Logistic Regression (With StandardScaler)
# -----------------------------

print("\n")
print("=" * 50)
print("Logistic Regression (With StandardScaler)")
print("=" * 50)

# Create StandardScaler
scaler = StandardScaler()

# Scale training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Logistic Regression model
lr_scaled = LogisticRegression(max_iter=1000)

# Train model
lr_scaled.fit(X_train_scaled, y_train)

# Prediction
y_pred_scaled = lr_scaled.predict(X_test_scaled)

print("\nAccuracy :", accuracy_score(y_test, y_pred_scaled))
print("Precision:", precision_score(y_test, y_pred_scaled))
print("Recall   :", recall_score(y_test, y_pred_scaled))
print("F1 Score :", f1_score(y_test, y_pred_scaled))

print("\nClassification Report")
print(classification_report(y_test, y_pred_scaled))

cm_scaled = confusion_matrix(y_test, y_pred_scaled)

print("\nConfusion Matrix")
print(cm_scaled)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_scaled,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression (With Scaling)")

plt.show()

print("\n")
print("=" * 50)
print("Comparison")
print("=" * 50)

print("Without Scaling Accuracy :", accuracy_score(y_test, y_pred))
print("With Scaling Accuracy    :", accuracy_score(y_test, y_pred_scaled))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# KNN Model
# -----------------------------

print("\n")
print("=" * 50)
print("K-Nearest Neighbors (KNN)")
print("=" * 50)

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("Accuracy :", knn_accuracy)
print("Precision:", precision_score(y_test, knn_pred))
print("Recall   :", recall_score(y_test, knn_pred))
print("F1 Score :", f1_score(y_test, knn_pred))

print("\nClassification Report")
print(classification_report(y_test, knn_pred))

# -----------------------------
# Decision Tree
# -----------------------------

print("\n")
print("=" * 50)
print("Decision Tree")
print("=" * 50)

dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("Accuracy :", dt_accuracy)
print("Precision:", precision_score(y_test, dt_pred))
print("Recall   :", recall_score(y_test, dt_pred))
print("F1 Score :", f1_score(y_test, dt_pred))

print("\nClassification Report")
print(classification_report(y_test, dt_pred))

# -----------------------------
# Model Comparison
# -----------------------------

comparison = {
    "Model": [
        "Logistic Regression (Scaled)",
        "KNN",
        "Decision Tree"
    ],
    "Accuracy": [
        accuracy_score(y_test, y_pred_scaled),
        knn_accuracy,
        dt_accuracy
    ]
}

comparison_df = pd.DataFrame(comparison)

print("\n")
print("=" * 50)
print("Model Comparison")
print("=" * 50)

print(comparison_df)

best_model = comparison_df.loc[
    comparison_df["Accuracy"].idxmax()
]

print("\nBest Model:")
print(best_model)

from sklearn.model_selection import GridSearchCV

# -----------------------------
# Hyperparameter Tuning for KNN
# -----------------------------

print("\n")
print("=" * 50)
print("Hyperparameter Tuning - KNN")
print("=" * 50)

# Parameter Grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Grid Search
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Train GridSearch
grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation Accuracy:")
print(grid_search.best_score_)

# Best Model
best_knn = grid_search.best_estimator_

# Prediction
best_pred = best_knn.predict(X_test_scaled)

print("\n")
print("=" * 50)
print("Best KNN Model Evaluation")
print("=" * 50)

print("Accuracy :", accuracy_score(y_test, best_pred))
print("Precision:", precision_score(y_test, best_pred))
print("Recall   :", recall_score(y_test, best_pred))
print("F1 Score :", f1_score(y_test, best_pred))

print("\nClassification Report")
print(classification_report(y_test, best_pred))

cm_best = confusion_matrix(y_test, best_pred)

print("\nConfusion Matrix")
print(cm_best)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_best,
    annot=True,
    fmt="d",
    cmap="Purples"
)

plt.title("Best KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

import joblib

joblib.dump(best_knn, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nBest KNN model saved successfully!")

# -----------------------------
# Feature Importance Analysis
# -----------------------------

print("\n")
print("=" * 50)
print("Feature Importance Analysis")
print("=" * 50)

feature_model = DecisionTreeClassifier(random_state=42)

feature_model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance")

plt.show()

