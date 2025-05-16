from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN
import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn  # for sklearn model tracking

def train_model(data_path, model_save_path):
    # Load dataset
    data = pd.read_csv(data_path)

    # Preprocess
    data.drop_duplicates(inplace=True)
    label_encoder = LabelEncoder()
    if 'GENDER' in data.columns:
        data['GENDER'] = label_encoder.fit_transform(data['GENDER'])
    if 'LUNG_CANCER' in data.columns:
        data['LUNG_CANCER'] = label_encoder.fit_transform(data['LUNG_CANCER'])
    data.replace({'YES': 2, 'NO': 1}, inplace=True)

    irrelevant_features = ['GENDER', 'AGE', 'SMOKING', 'SHORTNESS OF BREATH']
    df_new = data.drop(columns=irrelevant_features)
    df_new['ANXYELFIN'] = df_new['ANXIETY'] * df_new['YELLOW_FINGERS']

    # Oversample
    X = df_new.drop('LUNG_CANCER', axis=1)
    y = df_new['LUNG_CANCER']
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )

    gboost_params = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }

    # === MLflow Tracking ===
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Optional for local
    mlflow.set_experiment("Lung Cancer Prediction")

    with mlflow.start_run():
        grid_search = GridSearchCV(
            estimator=GradientBoostingClassifier(random_state=42),
            param_grid=gboost_params,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        joblib.dump(best_model, model_save_path)

        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

        # Log the model itself
        mlflow.sklearn.log_model(best_model, "model")

    return {
        "accuracy": grid_search.best_score_,
        "best_params": grid_search.best_params_
    }
