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

def scale_prediction(model_prediction, scale_size):
  # Adjust predicted data to scale
  adjusted_prediction = np.zeros(len(model_prediction))
  previous = 1
  for i in range(len(adjusted_prediction)):
    print(scale_size)
    print(model_prediction[i][0])
    adjusted_prediction[i] =  round(scale_size * model_prediction[i][0] * previous) % scale_size
    previous = adjusted_prediction[i]
    
  return adjusted_prediction.tolist()


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
    
    scale_size = int(first_eeg_snapshot[scale_index])

    total_features = len(first_eeg_snapshot)

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
    scaled_predictions = scale_prediction(predictions.tolist(), scale_size)
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
