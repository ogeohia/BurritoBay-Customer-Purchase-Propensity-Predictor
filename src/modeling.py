# -*- coding: utf-8 -*-

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
# No longer building a complex pipeline with custom transformers here,
# instead we save individual components for the prediction script.
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.cluster import KMeans

# Import preprocessing functions and resampling
from data_preprocessing import (
    load_data,
    handle_missing_values,
    perform_kmeans_clustering,
    encode_features,
    split_data, # Changed from split_and_resample_data
    resample_training_data
)

# Define file paths (relative to src/)
DATA_FILE_PATH = '../data/online_shoppers_intention.csv'
FITTED_KMEANS_MODEL_PATH = '../models/fitted_kmeans_model.pkl'
KMEANS_INPUT_PREPROCESSOR_PATH = '../models/kmeans_input_preprocessor.pkl'
FITTED_LABEL_ENCODERS_PATH = '../models/fitted_label_encoders.pkl'
LIGHTGBM_MODEL_PATH = '../models/lightgbm_model.pkl'
RANDOM_FOREST_MODEL_PATH = '../models/random_forest_model.pkl'
SELECTED_FEATURES_LIST_PATH = '../models/selected_features.pkl' # To save the list of features used

def train_model(X_train, y_train, model_type='lightgbm', random_state=42):
    """
    Trains a specified model on the training data.
    """
    print(f"\nTraining {model_type} model...")
    if model_type == 'lightgbm':
        # Use force_row_wise=True to suppress a warning with small datasets/specific configurations
        model = lgb.LGBMClassifier(random_state=random_state, force_row_wise=True)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    print(f"{model_type} model training complete.")
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the trained model and prints key metrics.
    """
    print(f"\nEvaluating {model_name}...")
    if model is None:
        print(f"Error: {model_name} is None. Cannot evaluate.")
        return

    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(f"{model_name} Accuracy:", accuracy_score(y_test, y_pred))
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
        print(f"{model_name} ROC-AUC Score:", roc_auc_score(y_test, y_prob))
        print("-" * 30)
    except Exception as e:
        print(f"Error during evaluation of {model_name}: {e}")


def save_artifact(artifact, file_path):
    """Saves a Python object (model, encoder, list, etc.) to a file using joblib."""
    try:
        joblib.dump(artifact, file_path)
        print(f"Artifact successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving artifact to {file_path}: {e}")


def load_artifact(file_path):
    """Loads a Python object from a file using joblib."""
    try:
        artifact = joblib.load(file_path)
        print(f"Artifact successfully loaded from {file_path}")
        return artifact
    except FileNotFoundError:
        print(f"Error: Artifact file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading artifact from {file_path}: {e}")
        return None


# Main execution block for the modeling script
if __name__ == '__main__':
    print("Starting modeling process...")

    # --- Data Loading and Preprocessing ---
    # We call the preprocessing functions directly here to get the prepared data
    data = load_data(DATA_FILE_PATH)

    if data is not None:
        data = handle_missing_values(data)

        # Perform K-Means clustering and get necessary artifacts
        data_with_cluster, fitted_kmeans_model, kmeans_input_preprocessor, kmeans_feature_names = perform_kmeans_clustering(data)

        if all(v is not None for v in [data_with_cluster, fitted_kmeans_model, kmeans_input_preprocessor, kmeans_feature_names]):

            # Separate features and target from the dataframe that now includes 'Cluster'
            X = data_with_cluster.drop('Revenue', axis=1)
            y = data_with_cluster['Revenue']

            # Split data into training and testing sets BEFORE final encoding
            X_train_raw_cluster, X_test_raw_cluster, y_train, y_test = split_data(X, y)

            if all(v is not None for v in [X_train_raw_cluster, X_test_raw_cluster, y_train, y_test]):

                # Encode categorical features on the training set and get the fitted encoders
                # This step is done AFTER splitting to prevent data leakage
                X_train_encoded, fitted_label_encoders = encode_features(X_train_raw_cluster)

                if X_train_encoded is not None and fitted_label_encoders is not None:
                    # Transform the test set using the encoders fitted on the training set
                    X_test_encoded, _ = encode_features(X_test_raw_cluster, fitted_label_encoders)

                    if X_test_encoded is not None:
                        # Apply resampling ONLY to the encoded training data
                        X_train_resampled_encoded, y_train_resampled = resample_training_data(X_train_encoded, y_train)

                        if X_train_resampled_encoded is not None and y_train_resampled is not None:
                            print("\nData prepared for modeling.")
                            print("X_train_resampled_encoded shape:", X_train_resampled_encoded.shape)
                            print("X_test_encoded shape:", X_test_encoded.shape)
                            print("y_train_resampled shape:", y_train_resampled.shape)
                            print("y_test shape:", y_test.shape)

                            # --- Feature Selection ---
                            # Based on your notebook, you used RandomForest feature importances
                            # and selected top features. You would typically do this step here
                            # using X_train_encoded (before resampling, to reflect original feature distribution)
                            # to get the list of selected features.
                            # For simplicity in this script, let's just use all features available
                            # after encoding for now, and add a placeholder for selected features.

                            # Example placeholder for feature selection (using all features):
                            selected_features = X_train_resampled_encoded.columns.tolist()
                            print(f"\nUsing {len(selected_features)} features for training.")
                            # In a real scenario, you would run a feature importance step here
                            # and narrow down 'selected_features'.

                            X_train_selected = X_train_resampled_encoded[selected_features]
                            X_test_selected = X_test_encoded[selected_features] # Select the same features from the test set

                            # --- Model Training ---
                            # Train LightGBM
                            lgb_model = train_model(X_train_selected, y_train_resampled, model_type='lightgbm')

                            # Train Random Forest
                            rf_model = train_model(X_train_selected, y_train_resampled, model_type='random_forest')


                            # --- Model Evaluation ---
                            evaluate_model(lgb_model, X_test_selected, y_test, model_name='LightGBM')
                            evaluate_model(rf_model, X_test_selected, y_test, model_name='Random Forest')


                            # --- Saving Artifacts ---
                            print("\nSaving models and preprocessing artifacts...")
                            save_artifact(lgb_model, LIGHTGBM_MODEL_PATH)
                            save_artifact(rf_model, RANDOM_FOREST_MODEL_PATH)
                            save_artifact(fitted_kmeans_model, FITTED_KMEANS_MODEL_PATH)
                            save_artifact(kmeans_input_preprocessor, KMEANS_INPUT_PREPROCESSOR_PATH)
                            save_artifact(fitted_label_encoders, FITTED_LABEL_ENCODERS_PATH)
                            save_artifact(selected_features, SELECTED_FEATURES_LIST_PATH) # Save list of features used

                            print("\nModeling process complete.")
                        else:
                             print("Error: Resampling failed.")
                    else:
                         print("Error: Test data encoding failed.")
                else:
                    print("Error: Training data encoding failed.")
            else:
                 print("Error: Data splitting failed.")
        else:
            print("Error: K-Means clustering or its preceding steps failed.")
    else:
        print("Error: Data loading failed.")
