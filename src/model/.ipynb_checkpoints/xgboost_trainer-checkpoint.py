import os
import mlflow
import pickle
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay


class PreprocessingModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["preprocessor"], "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(context.artifacts["classifier"], "rb") as f:
            self.classifier = pickle.load(f)

    def predict(self, context, model_input):
        processed_input = self.preprocessor.transform(model_input)
        try:
            processed_input = pd.DataFrame(
                processed_input,
                columns=self.preprocessor.get_feature_names_out()
            )
        except AttributeError:
            processed_input = pd.DataFrame(processed_input)
        return self.classifier.predict(processed_input)

def train_xgboost_model(x_train, y_train, x_test, y_test, preprocessor):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        early_stopping_rounds=10,
        verbosity=0 
    )

    param_grid = { 
        'n_estimators': [100, 200, 300], 
        'max_depth': [3, 4, 5], 
        'learning_rate': [0.01, 0.05, 0.1, 0.2], 
        'subsample': [0.7, 0.8, 0.9, 1.0], 
        'colsample_bytree': [0.8, 0.9, 1.0], 
        'gamma': [0, 0.1, 0.2] 
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='average_precision',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=1,
        n_jobs=-1
    )

    with mlflow.start_run(run_name="XGBoost_GridSearch"):
        grid_search.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        best_model = grid_search.best_estimator_

        y_proba = best_model.predict_proba(x_test)[:, 1]
        test_aucpr = average_precision_score(y_test, y_proba)
        mlflow.log_metric("test_aucpr", test_aucpr)

        log_plots(y_test, y_proba, best_model)

        artifacts = create_artifacts(preprocessor, best_model)
        
        input_example = pd.DataFrame({
            "Age": [30],
            "Job": [1],
            "Credit amount": [1000],
            "Duration": [12],
            "Sex": ["male"],
            "Housing": ["own"],
            "Saving accounts": ["little"],
            "Checking account": ["rich"],
            "Purpose": ["car"]
        })
        
        # Model signature inference
        signature = mlflow.models.infer_signature(
            input_example, 
            best_model.predict(preprocessor.transform(input_example))
        )

        mlflow.pyfunc.log_model(
            artifact_path="model_with_preprocessing",
            python_model=PreprocessingModel(),
            artifacts=artifacts,
            input_example=input_example,
            signature=signature
        )

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])

def log_plots(y_test, y_proba, model):
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax_pr)
    ax_pr.set_title("Precision-Recall Curve (Test Set)")
    mlflow.log_figure(fig_pr, "plots/precision_recall_curve.png")
    plt.close(fig_pr)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax_fi, importance_type='weight')
    ax_fi.set_title("Feature Importance (Weight)")
    plt.tight_layout()
    mlflow.log_figure(fig_fi, "plots/feature_importance.png")
    plt.close(fig_fi)

def create_artifacts(preprocessor, model):
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fp_pre:
        pickle.dump(preprocessor, fp_pre)
        preprocessor_path = fp_pre.name

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fp_clf:
        pickle.dump(model, fp_clf)
        classifier_path = fp_clf.name

    return {
        "preprocessor": preprocessor_path,
        "classifier": classifier_path
    }