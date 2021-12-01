from flask import Flask, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import os
from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)
model = keras.models.load_model("lstm_model_4.h5")
scale_index = 5
n_features = 8

def scale_prediction(model_prediction, scaler):
  # Adjust predicted data to scale
  num_notes_scale_index = scale_index
  inverse_scaler_dataset = np.zeros(shape=(len(model_prediction), n_features))
  inverse_scaler_dataset[:num_notes_scale_index] = model_prediction[:0]
  scaled_prediction = scaler.inverse_transform(inverse_scaler_dataset)[:0]

  adjusted_prediction = np.zeros(len(scaled_prediction))
  for i in range(len(test_data_scale_col)):
    scale_size = test_data_scale_col[i]
    scaled_adjustment = scaled_prediction[i] * 100
    adjusted_prediction[i] =  (scale_size * scaled_adjustment) % scale_size

  return adjusted_prediction


@app.post('/predict')
def predict():
    data = request.json['input']
    print(f"Input {data}")
    # 1. Create numpy array, splitting the giant string by the new line character.
    input_data_numpy_array = np.array(data.split("_,"))
    print(f"Input np array {input_data_numpy_array}")

    # 2. Determine the size of the input array for the ML model
    total_eeg_snapshots = len(input_data_numpy_array)
    first_eeg_snapshot = np.fromstring(
        input_data_numpy_array[0], dtype=float, sep=',')
    print(f"first_eeg_snapshot {first_eeg_snapshot}")

    total_features = len(first_eeg_snapshot)
    # numNotesInScaleColIndex = total_features - 2

    # 3. Create the array, build it up with the data!
    array = np.empty([total_eeg_snapshots, total_features])
    # numNotesInScaleCol = []
    for i in range(total_eeg_snapshots):
        array[i] = np.fromstring(
            input_data_numpy_array[i], dtype=float, sep=',')
        # numNotesInScaleCol.push(array[i][numNotesInScaleColIndex])
        # np.vstack([array, cur_eeg_snapshot_data])

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    array = scaler.fit_transform(array) 
    
    # 4. Predict!
    predictions = model.predict(array)
    # predictions = scale_prediction(predictions, numNotesInScaleCol)
    scaled_predictions = scale_prediction(predictions.tolist(), scaler)
    output_data = {"output": scaled_predictions}

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
