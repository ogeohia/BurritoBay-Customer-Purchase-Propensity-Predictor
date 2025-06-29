# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import sys
import os

# Add the parent directory (src/deployment) to the Python path
# This allows importing the 'predict' module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now you can import the predict module
try:
    from predict import predict_purchase_propensity
except ImportError as e:
    print(f"Error importing predict.py: {e}")
    print("Please ensure predict.py is in the directory:", parent_dir)
    # Exit or handle this error appropriately in a real application
    # For now, we'll allow the app to start but the /predict endpoint will fail
    predict_purchase_propensity = None


app = Flask(__name__)

@app.route('/')
def home():
    """Simple home endpoint to check if the API is running."""
    return "Purchase Propensity Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive session data and return purchase propensity prediction.
    Expects JSON data in the request body representing a single session.
    """
    if predict_purchase_propensity is None:
        return jsonify({"error": "Prediction module not loaded."}), 500

    if not request.json:
        return jsonify({"error": "Invalid input. Please provide JSON data."}), 400

    # Get the input data from the JSON request body
    input_data = request.json

    # Validate input_data structure if necessary (optional but recommended)
    # Ensure all expected keys are present, data types are correct, etc.
    # For simplicity here, we pass the raw dictionary to predict_purchase_propensity

    print(f"Received prediction request with data: {input_data}")

    # Call the predict_purchase_propensity function
    prediction_result = predict_purchase_propensity(input_data)

    # Return the prediction result as a JSON response
    # predict_purchase_propensity is designed to return a dictionary like
    # {"purchase_propensity": value} or {"error": message}
    if "error" in prediction_result:
         # If the prediction function returned an error, return it with a 500 status
         return jsonify(prediction_result), 500
    else:
         # Otherwise, return the prediction result with a 200 status
         return jsonify(prediction_result), 200


# This part makes the script runnable directly
if __name__ == '__main__':
    # Flask development server (for testing).
    # In production, you would use a WSGI server like Gunicorn or uWSGI.
    print("Starting Flask API...")
    app.run(debug=True, host='0.0.0.0', port=5000) # debug=True for development
