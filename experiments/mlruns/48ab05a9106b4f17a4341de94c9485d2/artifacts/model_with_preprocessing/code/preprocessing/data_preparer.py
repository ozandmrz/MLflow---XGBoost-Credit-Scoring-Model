import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .transformers import RareCategoryCombiner

class ModelDataPreparer:
    def __init__(self, key_vars: list[str] = [], seed: int = 42):
        self.key_vars = key_vars
        self.seed = seed
        self.preprocessor = None

    def prepare_and_split_data(self, df: pd.DataFrame, numeric_features, categorical_features, test_size: float = 0.2):
        df['Risk'] = df['Risk'].map({'good': 1, 'bad': 0})
        X = df.drop(columns=['Risk'] + self.key_vars)
        y = df['Risk']

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )

        self.preprocessor = self.create_preprocessor(numeric_features, categorical_features)
        processed_x_train = pd.DataFrame(
            self.preprocessor.fit_transform(x_train),
            columns=self.preprocessor.get_feature_names_out()
        )
        processed_x_test = pd.DataFrame(
            self.preprocessor.transform(x_test),
            columns=self.preprocessor.get_feature_names_out()
        )

        self.log_dataset_stats(processed_x_train, y_train, "train")
        self.log_dataset_stats(processed_x_test, y_test, "test")

        return processed_x_train, processed_x_test, y_train, y_test

    def create_preprocessor(self, numeric_features, categorical_features):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('rare_encoder', RareCategoryCombiner(threshold=0.05)),
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

    @staticmethod
    def log_dataset_stats(X: pd.DataFrame, y: pd.Series, dataset_name: str):
        stats = {
            'num_samples': X.shape[0],
            'num_features': X.shape[1],
            'class_balance': y.mean()
        }
        with mlflow.start_run(nested=True):
            mlflow.log_params({f'{dataset_name}_{k}': v for k, v in stats.items()})