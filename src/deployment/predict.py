# -*- coding: utf-8 -*-

import pandas as pd
import joblib
import numpy as np # Needed for array handling if preprocessors return arrays

# Define paths to the saved artifacts (relative to src/deployment/)
# Adjust these paths if your models/ directory is structured differently
FITTED_KMEANS_MODEL_PATH = '../../models/fitted_kmeans_model.pkl'
KMEANS_INPUT_PREPROCESSOR_PATH = '../../models/kmeans_input_preprocessor.pkl'
FITTED_LABEL_ENCODERS_PATH = '../../models/fitted_label_encoders.pkl'
# Choose which trained model to load (LightGBM or Random Forest)
TRAINED_MODEL_PATH = '../../models/lightgbm_model.pkl'
SELECTED_FEATURES_LIST_PATH = '../../models/selected_features.pkl'

# --- Load Saved Artifacts ---
print("Loading saved artifacts...")
fitted_kmeans_model = joblib.load(FITTED_KMEANS_MODEL_PATH)
kmeans_input_preprocessor = joblib.load(KMEANS_INPUT_PREPROCESSOR_PATH)
fitted_label_encoders = joblib.load(FITTED_LABEL_ENCODERS_PATH)
trained_model = joblib.load(TRAINED_MODEL_PATH)
selected_features = joblib.load(SELECTED_FEATURES_LIST_PATH)

print("Artifacts loaded successfully.")
print(f"Using model: {TRAINED_MODEL_PATH}")
print(f"Using {len(selected_features)} selected features.")


def preprocess_new_input(input_data, fitted_kmeans_model, kmeans_input_preprocessor, fitted_label_encoders, selected_features):
    """
    Applies the same preprocessing steps to a single new input data point
    as were applied to the training data (clustering, encoding, feature selection).
    """
    # Convert input_data (e.g., a dictionary) into a pandas DataFrame
    # Ensure it has the same columns as the original raw data used for training,
    # even if some values are missing in the input. The order is also important
    # for consistent preprocessing.
    # You might want to define a list of expected columns based on your original data.
    expected_columns = [
        'Administrative', 'Administrative_Duration', 'Informational',
        'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
        'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType',
        'Weekend'
    ]
    input_df = pd.DataFrame([input_data], columns=expected_columns)

    # Ensure data types are consistent if needed (e.g., handle boolean 'Weekend')
    # input_df['Weekend'] = input_df['Weekend'].astype(bool)
    # Handle numerical columns that might be strings
    # for col in ['Administrative', 'Administrative_Duration', ...]:
    #     if col in input_df.columns:
    #          input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0) # Example: coerce errors and fill NaN


    # --- Step 1: Add Cluster Feature ---
    # Apply the KMeans input preprocessor to get data in the format KMeans expects
    # Need to handle potential new categories gracefully with handle_unknown='ignore'
    # in the KMeans input preprocessor definition in data_preprocessing.py
    try:
        X_for_kmeans = kmeans_input_preprocessor.transform(input_df)

        # Ensure output is dense for KMeans prediction if it's sparse
        if hasattr(X_for_kmeans, 'toarray'):
             X_for_kmeans = X_for_kmeans.toarray()

        # Predict the cluster label using the fitted KMeans model
        cluster_label = fitted_kmeans_model.predict(X_for_kmeans)[0] # [0] because it's a single input

        # Add the cluster label as a new column to the input DataFrame
        input_df['Cluster'] = cluster_label

    except Exception as e:
        print(f"Error during KMeans clustering step in preprocessing: {e}")
        # Decide how to handle: e.g., assign a default cluster or return error
        # For now, let's re-raise or return None if a critical error occurs
        return None # Indicate preprocessing failure


    # --- Step 2: Encode Features (including Cluster) ---
    # Use the fitted Label Encoders to transform the relevant columns.
    X_encoded = input_df.copy()

    # Identify columns to encode: original object/bool columns + 'Cluster'
    cols_to_encode = X_encoded.select_dtypes(include=['object', 'bool']).columns.tolist()
    if 'Cluster' in X_encoded.columns and 'Cluster' not in cols_to_encode:
         cols_to_encode.append('Cluster')

    # Ensure columns to encode actually exist in the DataFrame
    cols_to_encode_present = [col for col in cols_to_encode if col in X_encoded.columns]


    for col in cols_to_encode_present:
        if col in fitted_label_encoders:
            # Use the fitted encoder
            le = fitted_label_encoders[col]
            # Transform, handling unseen labels. A robust way is needed here.
            # If the encoder was fitted with handle_unknown='ignore' (not default for LabelEncoder),
            # it might work. Otherwise, you need to handle ValueErrors.
            try:
                 # Ensure input data for the column is string type before transforming
                 X_encoded[col] = le.transform(X_encoded[col].astype(str))
            except ValueError as e:
                 print(f"Warning: Unseen label '{input_df[col].iloc[0]}' encountered during encoding for column '{col}'. Error: {e}")
                 # In a real scenario, you need a strategy:
                 # - Replace with a placeholder value (e.g., -1) if the model can handle it.
                 # - Map to a specific 'unknown' category if you added one during training.
                 # - Re-fit the encoder temporarily (less ideal).
                 # - Use OneHotEncoder with handle_unknown='ignore' in the first place.

                 # For this example, let's assign a placeholder (-1) if an unseen label occurs
                 # This requires the model trained on data where unseen labels were also handled this way.
                 print(f"Assigning -1 to unseen label in column '{col}'.")
                 # This replacement logic needs to be robust for different types of inputs
                 # For simplicity here, assuming a single value input:
                 try:
                      if input_df[col].iloc[0] not in le.classes_:
                           X_encoded[col] = -1 # Placeholder for unseen label
                      else:
                           # This case should not be reached if the ValueError was triggered
                           # but keeping for logical completeness.
                           X_encoded[col] = le.transform([input_df[col].iloc[0]].astype(str))[0]
                 except Exception as replace_e:
                      print(f"Error replacing unseen label in column '{col}': {replace_e}")
                      return None # Indicate preprocessing failure


        else:
            print(f"Warning: No fitted encoder found for column '{col}'. This column might not have been present or encoded during training.")
            # Decide how to handle: drop the column, assign a default value, etc.
            # For now, let's keep it as is, assuming it might be a numerical column not requiring encoding.


    # --- Step 3: Select Features ---
    # Ensure the encoded input DataFrame only contains the features the model was trained on.
    # This handles cases where the input might have extra columns or is missing expected ones.
    X_processed = X_encoded[selected_features]

    # Ensure the order of columns in X_processed matches the order the model expects.
    # The `selected_features` list loaded from the model artifacts should preserve this order.
    X_processed = X_processed[selected_features]


    return X_processed

def predict_purchase_propensity(input_data):
    """
    Loads artifacts (if not already loaded) and makes a purchase propensity prediction
    for a single input data point.
    """
    # Check if artifacts are loaded (handle potential loading errors at startup)
    if not all([fitted_kmeans_model, kmeans_input_preprocessor, fitted_label_encoders, trained_model, selected_features]):
        # Attempt to reload if needed, or return an error if loading failed initially
        print("Attempting to reload artifacts...")
        global fitted_kmeans_model, kmeans_input_preprocessor, fitted_label_encoders, trained_model, selected_features
        try:
            fitted_kmeans_model = joblib.load(FITTED_KMEANS_MODEL_PATH)
            kmeans_input_preprocessor = joblib.load(KMEANS_INPUT_PREPROCESSOR_PATH)
            fitted_label_encoders = joblib.load(FITTED_LABEL_ENCODERS_PATH)
            trained_model = joblib.load(TRAINED_MODEL_PATH)
            selected_features = joblib.load(SELECTED_FEATURES_LIST_PATH)
            print("Artifacts reloaded successfully.")
        except Exception as e:
            print(f"Error reloading artifacts: {e}")
            return {"error": "Model artifacts could not be loaded. Cannot make prediction."}


    try:
        # Apply preprocessing to the new input data
        processed_input = preprocess_new_input(
            input_data,
            fitted_kmeans_model,
            kmeans_input_preprocessor,
            fitted_label_encoders,
            selected_features
        )

        if processed_input is None:
             return {"error": "Preprocessing failed. Cannot make prediction."}


        # Ensure processed_input has the correct shape (1, n_features) for prediction
        if processed_input.shape[0] != 1:
            print(f"Warning: Processed input shape is {processed_input.shape}, expected (1, n_features).")
            # Attempt to reshape if it's a single row but not shaped correctly
            if len(processed_input.shape) == 1:
                 processed_input = processed_input.values.reshape(1, -1) # Reshape a single row Series/array

        # Make prediction using the trained model
        # Get the predicted probability of the positive class (Revenue = True)
        # predict_proba returns probabilities for both classes [prob_false, prob_true]
        prediction_proba = trained_model.predict_proba(processed_input)[:, 1]
        purchase_propensity = float(prediction_proba[0]) # Get the probability and convert to float

        # You can also get the predicted class label if needed
        # predicted_class = trained_model.predict(processed_input)[0]

        return {"purchase_propensity": purchase_propensity}

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Log the error and the input data for debugging
        print(f"Input data that caused error: {input_data}")
        return {"error": f"Prediction failed: {e}"}

# Example Usage (for testing the script directly)
if __name__ == '__main__':
    print("Running prediction script directly for testing...")

    # This example input should mimic the structure of a single row
    # of your original dataframe (excluding the 'Revenue' column).
    # Use values that are likely to exist in your training data categories.
    example_input_low_propensity = {
        'Administrative': 0,
        'Administrative_Duration': 0.0,
        'Informational': 0,
        'Informational_Duration': 0.0,
        'ProductRelated': 1,
        'ProductRelated_Duration': 0.0,
        'BounceRates': 0.20,
        'ExitRates': 0.20,
        'PageValues': 0.0,
        'SpecialDay': 0.0,
        'Month': 'Feb',
        'OperatingSystems': 1,
        'Browser': 1,
        'Region': 1,
        'TrafficType': 1,
        'VisitorType': 'Returning_Visitor',
        'Weekend': False
    }

    prediction_result_low = predict_purchase_propensity(example_input_low_propensity)
    print("\nExample Low Propensity Prediction Result:")
    print(prediction_result_low)

    # Example with values indicating potentially higher propensity (adjust based on your EDA insights)
    example_input_high_propensity = {
        'Administrative': 5,
        'Administrative_Duration': 100.0,
        'Informational': 2,
        'Informational_Duration': 50.0,
        'ProductRelated': 50,
        'ProductRelated_Duration': 1500.0,
        'BounceRates': 0.01,
        'ExitRates': 0.02,
        'PageValues': 25.0, # High PageValue
        'SpecialDay': 0.0,
        'Month': 'Nov', # High revenue month
        'OperatingSystems': 2,
        'Browser': 2,
        'Region': 3,
        'TrafficType': 2,
        'VisitorType': 'Returning_Visitor', # Or 'New_Visitor'
        'Weekend': False
    }

    prediction_result_high = predict_purchase_propensity(example_input_high_propensity)
    print("\nExample High Propensity Prediction Result:")
    print(prediction_result_high)

    # Example with an unseen categorical value (will trigger warning/error handling)
    # This assumes 'NewMonth' was not in the original training data's 'Month' column
    example_input_unseen_category = {
        'Administrative': 0,
        'Administrative_Duration': 0.0,
        'Informational': 0,
        'Informational_Duration': 0.0,
        'ProductRelated': 1,
        'ProductRelated_Duration': 0.0,
        'BounceRates': 0.20,
        'ExitRates': 0.20,
        'PageValues': 0.0,
        'SpecialDay': 0.0,
        'Month': 'NewMonth', # Unseen category
        'OperatingSystems': 1,
        'Browser': 1,
        'Region': 1,
        'TrafficType': 1,
        'VisitorType': 'Returning_Visitor',
        'Weekend': False
    }
    prediction_result_unseen = predict_purchase_propensity(example_input_unseen_category)
    print("\nExample Unseen Category Prediction Result:")
    print(prediction_result_unseen)
