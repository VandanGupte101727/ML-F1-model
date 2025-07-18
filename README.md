# ML-F1-model
machine learning model for Formula 1 datasets 

# 🏎️ F1 Fatal Accidents Analysis & Session Prediction with Machine Learning

This project explores and models historical data from fatal Formula 1 accidents to **predict the session type** (e.g., Practice, Qualifying, Race) using machine learning techniques. The primary model used is a **Random Forest Classifier**, applied to driver-related accident data.

---

## 📂 Dataset Overview

The project uses four CSV datasets:

1. **`fatal_accidents_drivers.csv`** — Driver accident details
2. **`fatal_accidents_marshalls.csv`** — Accidents involving marshalls
3. **`red_flags.csv`** — Red flag incidents and resumptions
4. **`safety_cars.csv`** — Safety car deployments and reasons

---

## 🧠 Objective

- Preprocess and clean the accident datasets
- Encode categorical features and engineer new features
- Train a classifier to **predict the session** (e.g., Race, Practice) based on accident data
- Evaluate the model using accuracy, confusion matrix, and classification report

---

## 🛠️ Technologies Used

- Python (Google Colab)
- Pandas & NumPy
- Matplotlib & Seaborn (visualization)
- Scikit-learn (ML modeling & preprocessing)

---

## 📊 Feature Engineering

- Label encoding of `Driver`, `Event`, `Car`, and `Session`
- Computed ratios like:
  - `Session_ratio = Session / Driver`
  - `Driver_age_ratio = Driver / Age`
- One-hot encoding of categorical columns
- Handling missing and infinite values

---

## 🤖 Model

- **RandomForestClassifier**
  - `class_weight="balanced"` to account for class imbalance
  - Trained on 80% of filtered dataset using **StratifiedShuffleSplit**
  - Accuracy: ~55% (subject to improvement with more features)

---

## ✅ Evaluation Metrics

- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**
- **Prediction vs Actual Comparison (Top 10 rows)**

Example:

| Actual Session | Predicted Session |
|----------------|-------------------|
| 1              | 4                 |
| 4              | 4                 |
| 2              | 1                 |

---

## 📈 Suggestions for Improvement

- Add more predictive features (weather, team data, circuit type)
- Try other models (e.g., XGBoost, CatBoost)
- Apply advanced sampling techniques (SMOTE, undersampling)
- Map encoded session values back to readable labels

---

## 🗂️ Folder Structure


