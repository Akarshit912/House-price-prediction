import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load the dataset and model
data = pd.read_csv(r".\House_Data.csv")
model = pickle.load(open(r".\home_prices_model.pickle", "rb"))

# Load the columns information from the columns.json file
with open(r".\columns.json", "r") as f:
    columns_info = json.load(f)
    availability_values = columns_info["availability_columns"]
    area_values = columns_info["area_columns"]
    location_values = columns_info["location_columns"]

# Print column counts for debugging
print(f"Location columns: {len(location_values)}")
print(f"Availability columns: {len(availability_values)}")
print(f"Area columns: {len(area_values)}")

@app.route('/')
def index():
    locations = data['location'].unique()
    return render_template('index.htm', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs
    location = request.form.get('location')
    bhk = int(request.form.get('bhk'))  # Convert BHK to integer
    bath = int(request.form.get('bath'))  # Convert bathrooms to integer
    sqft = float(request.form.get('total_sqft'))  # Convert square footage to float

    # One-hot encoding of categorical values
    loc_array = np.zeros(len(location_values))
    if location in location_values:
        loc_array[location_values.index(location)] = 1

    # Handle other categorical features (availability and area)
    availability = "Ready To Move"  # Replace with actual value from form
    area = "Super built-up Area"  # Replace with actual value from form
    availability_array = np.zeros(len(availability_values))
    area_array = np.zeros(len(area_values))

    # Set the correct index for availability and area if they match
    if availability in availability_values:
        availability_array[availability_values.index(availability)] = 1
    if area in area_values:
        area_array[area_values.index(area)] = 1
    availability_array = availability_array[:-1]
    area_array = area_array[:-1]
    loc_array = loc_array[:-1]
    # Construct the feature array
    sample = np.concatenate(([sqft, bhk, bath], availability_array, area_array, loc_array))
    print(sample)
    # Ensure that the number of features matches the model's expectation
    print(f"{sample.shape}")  # Check the sample shape

    if len(sample) != 248:
        return f"Error: The feature vector has {len(sample)} features, but the model expects 248."

    # Make the prediction
    prediction = model.predict(sample.reshape(1, -1))[0]  # Reshape for prediction

    return f" {prediction*10000:,.2f}"

if __name__ == "__main__":
    app.run(debug=True, port=5001)
