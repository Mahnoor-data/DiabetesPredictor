# Diabetes Prediction — End-to-End Machine Learning Project

> Predicting diabetes from patient medical records using machine learning, with a live interactive demo deployed on Hugging Face Spaces.

---

## Live Demo

**Try the app here:** [https://huggingface.co/spaces/noormrc123/diabetes-prediction]

---

## Demo

[

https://github.com/user-attachments/assets/000d9074-163f-4bc8-9bc9-3226beec2220

]

---

## Project Overview

This is a complete, production-style data science project built from scratch — from raw data exploration all the way to a deployed web application. The goal is to predict whether a patient has diabetes based on their medical and demographic information.

The dataset contains **100,000 patient records** sourced from Kaggle, with features like HbA1c level, blood glucose, BMI, age, and smoking history.

---

## Results at a Glance

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| Logistic Regression | 88.54%   | 0.9624  |
| XGBoost             | 97.23%   | 0.9968  |
| **Random Forest**   | **97.52%** | **0.9972** |

**Best Model: Random Forest — 97.52% Accuracy, ROC-AUC 0.9972**

---

## Visualizations

### Target Variable Distribution
[<img width="512" height="347" alt="image" src="https://github.com/user-attachments/assets/8652f8f8-9c2b-4070-b672-eca47d3c67db" />
]

### Feature Distributions
[<img width="457" height="347" alt="image" src="https://github.com/user-attachments/assets/2bd9a744-e022-470c-915c-0bd825ccf9c3" />
<img width="1022" height="360" alt="image" src="https://github.com/user-attachments/assets/f270a766-33b6-4fe5-99e7-47a2135166a4" />

]

### Correlation Heatmap
[<img width="492" height="442" alt="image" src="https://github.com/user-attachments/assets/15399f9a-adec-443b-8406-52316b6137d9" />
]

### Confusion Matrices
[<img width="1036" height="312" alt="image" src="https://github.com/user-attachments/assets/8f0acc92-d9e1-4c13-b9ae-a3f841e5536c" />
]

### ROC Curves
[<img width="357" height="300" alt="image" src="https://github.com/user-attachments/assets/b1241ea8-b7fe-4a8b-813e-35bee31d42f9" />
]

### Feature Importance
[<img width="1002" height="580" alt="image" src="https://github.com/user-attachments/assets/abf5e2af-a827-4907-af5e-be35bcd1d249" />
]

---

## Feature Importance — What Actually Predicts Diabetes?

```
1. HbA1c_level            ########   0.4015   (40%)
2. blood_glucose_level    ######     0.2528   (25%)
3. age                    #####      0.1930   (19%)
4. bmi                    ###        0.1104   (11%)
5. smoking_history        #          0.0291    (3%)
6. gender                            0.0076    (1%)
7. hypertension                      0.0034    (0.3%)
8. heart_disease                     0.0022    (0.2%)
```

HbA1c and blood glucose together account for **65% of the prediction** — which aligns perfectly with how doctors diagnose diabetes in real clinical settings.

---

## Project Workflow

```
Data Loading
     |
Exploratory Data Analysis (EDA)
     |
Data Preprocessing
  - Label Encoding (gender, smoking_history)
  - SMOTE — fixed 10.76:1 class imbalance
  - StandardScaler — feature normalization
     |
Model Training
  - Logistic Regression
  - XGBoost
  - Random Forest  <-- best model
     |
Evaluation
  - Accuracy, ROC-AUC, Confusion Matrix, ROC Curve
     |
Deployment
  - Gradio app on Hugging Face Spaces
```

---

## Tech Stack

- **Language:** Python
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Imbalanced Data:** imbalanced-learn (SMOTE)
- **Deployment:** Gradio, Hugging Face Spaces
- **Notebook:** Google Colab

---

## Dataset

- **Source:** [Kaggle — Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Records:** 100,000 patients
- **Features:** 8 input features + 1 target column
- **Class Distribution:** 91,500 non-diabetic / 8,500 diabetic (before SMOTE)

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Mahnoor-data/DiabetesPredictor.git
cd diabetes-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

---

## Project Structure

```
diabetes-prediction/
|
├── notebook.ipynb          # Full Colab notebook with all steps
├── app.py                  # Gradio web app
├── requirements.txt        # Dependencies
├── diabetes_model.pkl      # Trained Random Forest model
├── scaler.pkl              # Fitted StandardScaler
└── README.md
```

---

## Key Takeaways

- A severe class imbalance of **10.76:1** was handled using SMOTE, bringing both classes to 91,500 samples each — this was critical for fair model training
- Random Forest significantly outperformed Logistic Regression, showing that the relationship between features and diabetes is **non-linear**
- The top two features alone (HbA1c + blood glucose) drive 65% of predictions, which is **medically validated**
- The project follows a **full production pipeline** — not just a notebook, but a deployed, usable application

---

## About

**Mahnoor**  Data Scientist

[https://www.linkedin.com/in/mahnoor-zakir-9a6183358] 
