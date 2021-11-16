from flask import Flask, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
model = keras.models.load_model("lstm_model_4.h5")


@app.post('/predict')
def predict():
    data = request.form['input']

    # 1. Create numpy array, splitting the giant string by the new line character.
    input_data_numpy_array = np.array(data.split("\n"))

    # 2. Determine the size of the input array for the ML model
    total_eeg_snapshots = len(input_data_numpy_array)
    first_snapshot = input_data_numpy_array[0].split(",")
    total_features = len(first_snapshot)

    # 3. Create the array, build it up with the data!
    array = np.empty([total_eeg_snapshots, total_features])
    for i in range(total_eeg_snapshots):
        array[i] = input_data_numpy_array[i].split(",")
        array[i] = float(array[i])
        # np.vstack([array, cur_eeg_snapshot_data])

    # 4. Predict!
    predictions = model.predict(array)
    output_data = {"output": predictions.tolist()}

    print(output_data)
    return output_data

    # Maybe add retraining? idk
    # Fit model and get prediction
    # model.fit(test_input, test_target)


# @app.get('/summary')
# def getModelSummary():
#     summary = model.summary()
#     return summary


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
