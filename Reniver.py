
"""
DATA pipeline for IOT data at RENIVER of a  langZuaner smart Hydarulic press
"""

from flask import Flask, request, jsonify
import numpy as NumericalPython # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

downloads_folder = r"C:\Users\ishaa\Downloads\archive" 
file_list = [os.path.join(downloads_folder, filename) for filename in os.listdir(downloads_folder) if len(filename) < 8]
"""
output
Cooler condition : 3: close to total failure 20: reduced effifiency 100: full efficiency
stable flag: 0: conditions were unstable  1: stable conditions
"""

# Create a Flask web app
app = Flask(__name__)
TARGET_NAMES = ["cooler", "valve", "leakage", "accumulator", "stable"]
def reading_from_file(file):
    return NumericalPython.genfromtxt(file, dtype=float, delimiter='\t')

Dictionary_data_values = {}
for file in file_list:
    data = reading_from_file(file)
    file_name = file.split('/')[-1]
    file_ = file_name.split('.')[0]
    print(file_, data.shape)
    Dictionary_data_values[file_] =  data
dataframe_target = pd.read_table(r'C:\Users\ishaa\Downloads\archive/profile.txt' , names=TARGET_NAMES)

unstables = NumericalPython.where(dataframe_target['stable'].values == 0)[0]
stables = NumericalPython.where(dataframe_target['stable'].values == 1)[0]
"""
The numpy.where() function is used in the provided code to find the indices of elements in an array that satisfy a particular condition
"""
num = 250
plt.figure(figsize = (16,16))
for i,key in enumerate(Dictionary_data_values.keys()):
    
    plt.subplot(4,4,i+1)
    plt.title(f'{key}')
    for v in stables[:num-1]:
        plt.plot(Dictionary_data_values[key][v], color='red')
    plt.plot(Dictionary_data_values[key][stables[num]], label='stable', color='blue')
    for v in unstables[:num]:
        plt.plot(Dictionary_data_values[key][v], color = 'blue', alpha=0.5)
    plt.plot(Dictionary_data_values[key][unstables[num]], label='unstable', color ='red', alpha=0.5)
    plt.legend(loc='upper right')
    plt.savefig(f'{key}.jpg')
    
def preprocess_data(Dictionary_data_values, window_size=1):
   
    x = None

    for key in Dictionary_data_values:
        v = Dictionary_data_values[key]
        if v.shape[1] == 6000:
            v = v[:,::100]
        elif v.shape[1] == 600:
            v = v[:,::10]
        assert v.shape[1] == 60

        v = NumericalPython.apply_along_axis(lambda x: NumericalPython.convolve(x, NumericalPython.ones(window_size)/window_size, mode='valid'), axis = 1, arr=v)
        scaler = MinMaxScaler()
        v = scaler.fit_transform(v)
        if x is None:
            x = v.reshape(1,2205,61 - window_size)
        else:
            x = NumericalPython.concatenate((x,v.reshape(1,2205,61 - window_size ))) # The shape is typically (num_samples, num_time_steps, num_features).
    x = NumericalPython.transpose(x,(1,2,0))
    return x


x = preprocess_data(Dictionary_data_values, window_size=5)
y = dataframe_target['stable'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)    
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Plot training loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss vs. Validation Loss')
plt.show()

# Evaluate the model on training and test data
train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}")
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
"""
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Receive data from the request

    # Implement data preprocessing for prediction
    preprocessed_data = preprocess_data(data)

    # Make predictions using your model
    predictions = model.predict(preprocessed_data)

    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
"""
