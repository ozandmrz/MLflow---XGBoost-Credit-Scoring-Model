import pandas as pd
import mlflow
from preprocessing.data_preparer import ModelDataPreparer
from model.xgboost_trainer import train_xgboost_model  


mlflow.set_tracking_uri("sqlite:///../experiments/credit_risk_model.db")
existing_exp = mlflow.get_experiment_by_name('CreditRiskModel')
if not existing_exp:
    mlflow.create_experiment('CreditRiskModel', artifact_location="../experiments/mlruns")
mlflow.set_experiment('Credit Score')
NUMERIC_FEATURES = ['Age', 'Job', 'Credit amount', 'Duration']
CATEGORICAL_FEATURES = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

if __name__ == "__main__":    
    df = pd.read_csv("../data/german_credit_data.csv")
    
    preparer = ModelDataPreparer()
    x_train, x_test, y_train, y_test = preparer.prepare_and_split_data(
        df, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )

    trained_pipeline = train_xgboost_model(
        x_train, y_train,
        x_test, y_test,
        preparer.preprocessor   )
    print("Training completed successfully!")