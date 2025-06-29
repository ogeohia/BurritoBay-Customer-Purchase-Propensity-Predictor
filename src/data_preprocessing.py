# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np # Often useful for numerical operations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from imblearn.over_sampling import RandomOverSampler
import joblib # To save fitted preprocessing components (KMeans model, encoders)

def load_data(file_path):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(df):
    """Checks and handles missing values (based on notebook, there are none)."""
    if df is None:
        return None
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found per column:\n", missing_values)
        # Add handling logic here if needed (e.g., imputation, dropping rows)
        # For now, based on the notebook, we just report.
    else:
        print("No missing values found.")
    return df

def perform_kmeans_clustering(df, n_clusters=3, random_state=42):
    """
    Applies K-Means clustering as a feature engineering step.
    Returns a tuple:
    - DataFrame with the new 'Cluster' column added.
    - The fitted KMeans model.
    - The ColumnTransformer used internally for KMeans input OHE (needed for prediction).
    - List of feature names used for clustering (output of the ColumnTransformer).
    """
    if df is None:
        return None, None, None, None

    df_copy = df.copy() # Work on a copy

    # Prepare data for clustering: One-Hot Encode original categorical features
    # Exclude the target variable 'Revenue' from features for clustering
    categorical_cols_for_ohe = df_copy.select_dtypes(include=['object', 'bool']).columns.tolist()
    if 'Revenue' in categorical_cols_for_ohe:
        categorical_cols_for_ohe.remove('Revenue')

    numerical_cols = df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Ensure numerical columns used for clustering exist in the dataframe copy
    numerical_cols_present = [col for col in numerical_cols if col in df_copy.columns]


    # Create a ColumnTransformer for One-Hot Encoding of original categorical features
    # Pass through numerical columns needed for clustering
    # remainder='passthrough' is used here to keep numerical columns that were not
    # explicitly specified but are needed for clustering input.
    preprocessor_for_clustering_input = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols_present),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_for_ohe)
        ],
        remainder='drop' # Drop any other columns not specified
    )

    try:
        # Fit and transform the data for KMeans input
        df_processed_array = preprocessor_for_clustering_input.fit_transform(df_copy)

        # Get feature names after transformation for clustering
        feature_names_for_clustering_input = []
        for name, transformer, original_cols in preprocessor_for_clustering_input.transformers_:
            if transformer == 'passthrough':
                feature_names_for_clustering_input.extend(original_cols)
            elif hasattr(transformer, 'get_feature_names_out'):
                feature_names_for_clustering_input.extend(transformer.get_feature_names_out(original_cols))
            # Handle remainder if not 'drop'

        # Check if df_processed_array is sparse and convert to dense if needed for KMeans
        if isinstance(df_processed_array, np.ndarray) or hasattr(df_processed_array, 'toarray'):
             df_features_for_clustering = pd.DataFrame(df_processed_array, columns=feature_names_for_clustering_input, index=df_copy.index)
             if hasattr(df_processed_array, 'toarray'): # If sparse
                  df_features_for_clustering = pd.DataFrame(df_processed_array.toarray(), columns=feature_names_for_clustering_input, index=df_copy.index)
        else:
             print("Warning: Unexpected output type from preprocessor_for_clustering_input.")
             df_features_for_clustering = pd.DataFrame(df_processed_array, columns=feature_names_for_clustering_input, index=df_copy.index)


        # Apply K-Means Clustering
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans_model.fit(df_features_for_clustering)

        # Add 'Cluster' feature to the input dataframe copy
        df_copy['Cluster'] = kmeans_model.labels_
        print(f"K-Means clustering performed with K={n_clusters}. 'Cluster' feature added.")

        # Return the DataFrame with the cluster feature, the fitted KMeans model,
        # and the fitted preprocessor used for KMeans input (needed to preprocess new data for clustering)
        return df_copy, kmeans_model, preprocessor_for_clustering_input, feature_names_for_clustering_input

    except Exception as e:
        print(f"Error during K-Means clustering: {e}")
        return df_copy, None, None, None

def encode_features(X, fitted_label_encoders=None):
    """
    Encodes categorical features (object, bool, and the 'Cluster' integer column).
    Can use pre-fitted encoders for transformation or fit new ones.
    Returns the encoded features and the dictionary of fitted encoders.
    """
    if X is None:
        return None, None

    X_encoded = X.copy()
    label_encoders = fitted_label_encoders if fitted_label_encoders is not None else {}

    # Identify columns to encode: original object/bool columns + the 'Cluster' column
    # Ensure 'Cluster' is treated as categorical for encoding purposes here
    cols_to_encode = X_encoded.select_dtypes(include=['object', 'bool']).columns.tolist()
    if 'Cluster' in X_encoded.columns and 'Cluster' not in cols_to_encode:
         cols_to_encode.append('Cluster')

    # Ensure columns to encode actually exist in the DataFrame
    cols_to_encode_present = [col for col in cols_to_encode if col in X_encoded.columns]


    for col in cols_to_encode_present:
         # Ensure values are treated as strings before encoding
        col_data_str = X_encoded[col].astype(str)

        if col in label_encoders:
            # Use existing fitted encoder for transformation
            try:
                X_encoded[col] = label_encoders[col].transform(col_data_str)
            except ValueError as e:
                 # Handle unseen labels during transformation (common in test/new data)
                 # In a production setting, you need a robust strategy:
                 # Option 1 (simple): Replace with a placeholder (e.g., -1) if the encoder was fitted with handle_unknown='error'
                 # Option 2: Ensure your encoding strategy in the pipeline handles unknowns (e.g., OneHotEncoder with handle_unknown='ignore')
                 # For LabelEncoder, the robust way is to catch the error and decide what to do.
                 print(f"Warning: Unseen label during transformation for column '{col}'. Error: {e}")
                 # A common workaround for prediction is to fit the encoder on the combined known and unknown data
                 # to include the new label, and then transform. However, this is not ideal.
                 # A better approach for deployment is to use OneHotEncoder with handle_unknown='ignore'
                 # within a pipeline or ensure all possible labels are seen during initial fitting.

                 # For this refactoring, let's simulate handling by setting a default or re-fitting if necessary.
                 # A safer approach might be to convert to OneHotEncoder for deployment if handle_unknown is important.
                 # For LabelEncoder, if transform fails, let's assign a placeholder (e.g., -1)
                 # Note: This requires careful consideration of model training - models might not handle -1 well.
                 print(f"Setting unseen labels in '{col}' to -1.")
                 # Identify unseen labels
                 unseen_labels = set(col_data_str) - set(label_encoders[col].classes_)
                 # Replace unseen labels
                 col_data_str_replaced = col_data_str.replace(list(unseen_labels), '-1') # This needs to be done carefully, maybe map unknown
                 # A robust way using mapping:
                 mapping = {label: idx for idx, label in enumerate(label_encoders[col].classes_)}
                 mapped_data = col_data_str.apply(lambda x: mapping.get(x, -1)) # Map known, assign -1 to unknown
                 X_encoded[col] = mapped_data

                 # If you strictly need LabelEncoder output for your model, and can't use -1,
                 # you might need to collect all possible categories during training and
                 # fit the encoder on all of them, or use a different strategy.

                 # For this example, we'll proceed assuming unseen labels might cause issues
                 # if the model expects specific label ranges. A robust pipeline is key.

        else:
            # Fit and store a new encoder if not provided (used during initial training)
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(col_data_str)
            label_encoders[col] = le # Store the new fitted encoder

    return X_encoded, label_encoders


def split_data(X, y, random_state=42, test_size=0.2):
    """
    Splits data into training and testing sets.
    """
    if X is None or y is None:
        return None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print("Data split into training and testing sets.")
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def resample_training_data(X_train, y_train, random_state=42):
     """
     Applies Random Over-sampling to the training set.
     """
     if X_train is None or y_train is None:
         return None, None

     print("Applying Random Over-sampling...")
     ros = RandomOverSampler(random_state=random_state)
     X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
     print(f"Resampled training shape: {X_train_resampled.shape}")
     print("Class distribution in resampled training data:\n", y_train_resampled.value_counts())

     return X_train_resampled, y_train_resampled

# Main execution block for testing the script
if __name__ == '__main__':
    # Example usage:
    # Update with your actual data file path (relative to src/)
    file_path = '../data/online_shoppers_intention.csv'

    # Load data
    data = load_data(file_path)

    if data is not None:
        # Handle missing values (checks based on notebook)
        data = handle_missing_values(data)

        # Perform K-Means clustering and get the fitted model and data with cluster feature
        data_with_cluster, fitted_kmeans_model, kmeans_input_preprocessor, kmeans_feature_names = perform_kmeans_clustering(data)

        if data_with_cluster is not None and fitted_kmeans_model is not None and kmeans_input_preprocessor is not None and kmeans_feature_names is not None:

            # Separate features and target from the dataframe that now includes 'Cluster'
            X = data_with_cluster.drop('Revenue', axis=1)
            y = data_with_cluster['Revenue']

            # Split data into training and testing sets
            X_train_raw_cluster, X_test_raw_cluster, y_train, y_test = split_data(X, y)

            if X_train_raw_cluster is not None:
                # Encode categorical features on the training set and get the fitted encoders
                X_train_encoded, fitted_label_encoders = encode_features(X_train_raw_cluster)

                if X_train_encoded is not None and fitted_label_encoders is not None:
                    # Transform the test set using the encoders fitted on the training set
                    X_test_encoded, _ = encode_features(X_test_raw_cluster, fitted_label_encoders)

                    if X_test_encoded is not None:
                        # Apply resampling to the encoded training data
                        X_train_resampled_encoded, y_train_resampled = resample_training_data(X_train_encoded, y_train)

                        if X_train_resampled_encoded is not None:
                            print("\nData preprocessing complete.")
                            print("Shapes after preprocessing and resampling:")
                            print("X_train_resampled_encoded:", X_train_resampled_encoded.shape)
                            print("X_test_encoded:", X_test_encoded.shape)
                            print("y_train_resampled:", y_train_resampled.shape)
                            print("y_test:", y_test.shape)
                            print("\nNumber of features after encoding:", X_train_resampled_encoded.shape[1])

                            # In a real workflow, you would save `fitted_kmeans_model`,
                            # `kmeans_input_preprocessor`, and `fitted_label_encoders` here
                            # to be used by the modeling and prediction scripts.
                            # Example saving:
                            # joblib.dump(fitted_kmeans_model, '../models/fitted_kmeans_model.pkl')
                            # joblib.dump(kmeans_input_preprocessor, '../models/kmeans_input_preprocessor.pkl')
                            # joblib.dump(fitted_label_encoders, '../models/fitted_label_encoders.pkl')
