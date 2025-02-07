import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class RareCategoryCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.rare_categories_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            counts = X[col].value_counts(normalize=True)
            self.rare_categories_[col] = list(counts[counts < self.threshold].index)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for col in X.columns:
            if col in self.rare_categories_:
                X[col] = np.where(X[col].isin(self.rare_categories_[col]), 'Other', X[col])
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(list(self.rare_categories_.keys()))
        return np.array(input_features)