# MLflow & XGBoost Credit Scoring Model

Welcome! This repository showcases a hands-on MLOps project where I integrated an XGBoost classifier with MLflow to build a reproducible and deployable credit scoring model. Using 1000 samples from the German Credit dataset, the project demonstrates how to streamline the entire machine learning workflow—from data preprocessing and hyperparameter tuning to model evaluation and logging.

---

## What’s Inside?

- **Hyperparameter Tuning:**  
  Leveraging GridSearchCV, the project finds the optimal parameters for the XGBoost model, ensuring you get the best performance possible.

- **MLflow Integration:**  
  Every experiment, metric, and plot is automatically logged using MLflow. This not only makes tracking your work a breeze but also helps in reproducing results effortlessly.

- **Preprocessing Pipeline:**  
  The code includes a dedicated data preprocessing module that prepares your data seamlessly before model training.

- **Comprehensive Evaluation:**  
  You’ll get detailed evaluation outputs, including precision-recall curves and feature importance visualizations, to better understand how your model is performing.

- **Production-Ready Packaging:**  
  With a custom MLflow model wrapper, this project is designed to be easily deployed into production environments.

---

## Technologies Used

- **XGBoost** – For building the gradient boosting classifier.
- **MLflow** – To track experiments, manage models, and log artifacts.
- **scikit-learn** – Providing the preprocessing utilities and hyperparameter tuning with GridSearchCV.
- **Pandas** – For efficient data manipulation.
- **Matplotlib** – To create insightful plots and visualizations.

---

## Project Structure

```
├── experiments/          # Contains MLflow tracking artifacts and metadata.
│   ├── mlruns/           # Default MLflow tracking directory.
│   └── credit_risk_model.db
├── data/                 # Sample data files.
│   └── german_credit_data.csv
├── src/                  # Main source code.
│   ├── preprocessing/    # Data preprocessing modules.
│   │   ├── transformers.py
│   │   └── data_preparer.py
│   ├── models/           # Model training modules.
│   │   └── xgboost_trainer.py
│   └── main.py           # The main script to run the project.
├── tests.py              # Test cases for the project.
├── requirements.txt      # List of project dependencies.
└── README.md             # Project documentation.
```

---

This project is my personal exploration into automating the ML model lifecycle. Whether you’re looking to understand how MLflow can enhance your experiments or you’re in need of a starting point for a production-ready credit scoring model, I hope you find this repository useful. Happy experimenting!
