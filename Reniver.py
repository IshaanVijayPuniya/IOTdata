
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
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping

downloads_folder = r"C:\Users\ishaa\Downloads\archive" 
file_list = [os.path.join(downloads_folder, filename) for filename in os.listdir(downloads_folder) if len(filename) < 8]
"""
output
Cooler condition : 3: close to total failure 20: reduced effifiency 100: full efficiency
stable flag: 0: conditions were unstable  1: stable conditions
"""
def build_train_model(data_folder, profile_file, window_size=5, lstm_units=64, dropout_rate=0.5, epochs=50, test_size=0.2):
    # Load data from files
    downloads_folder = data_folder
    file_list = [os.path.join(downloads_folder, filename) for filename in os.listdir(downloads_folder) if len(filename) < 8]

    def reading_from_file(file):
        return NumericalPython.genfromtxt(file, dtype=float, delimiter='\t')

    Dictionary_data_values = {}
    for file in file_list:
        data = reading_from_file(file)
        file_name = file.split('/')[-1] # Use backslash for Windows paths
        file_ = file_name.split('.')[0]
        print(file_, data.shape)
        Dictionary_data_values[file_] = data

    TARGET_NAMES = ["cooler", "valve", "leakage", "accumulator", "stable"]

    dataframe_target = pd.read_table(profile_file, names=TARGET_NAMES)
    
    """
    The numpy.where() function is used in the provided code to find the indices of elements in an array that satisfy a particular condition
    """
    unstables = NumericalPython.where(dataframe_target['stable'].values == 0)[0]
    stables = NumericalPython.where(dataframe_target['stable'].values == 1)[0]
    num = 250
    plt.figure(figsize = (32,32))
    for i,key in enumerate(Dictionary_data_values.keys()): # enumerate i means 0 1 2 3 4 here
        
        plt.subplot(5,5,i+1)
        plt.title(f'{key}')
        for v in stables[:num]:
            plt.plot(Dictionary_data_values[key][v], color='red')
        plt.plot(Dictionary_data_values[key][stables[num]], label='stable', color='blue')
        for v in unstables[:num]:
            plt.plot(Dictionary_data_values[key][v], color = 'blue', alpha=0.5)
        plt.plot(Dictionary_data_values[key][unstables[num]], label='unstable', color ='red', alpha=0.5)
        plt.legend(loc='upper right')
        plt.savefig(f'{key}.jpg')
    

    def preprocess_data(data_dict, size=1):
        x = None

        for key in data_dict:
            values = data_dict[key]
            if values.shape[1] == 6000:
                values = values[:, ::100] # STEPS of 100 to make 60 rows
            elif values.shape[1] == 600:
                values = values[:, ::10] # STEPS of 10 to make 60 rows
            assert values.shape[1] == 60

            values = NumericalPython.apply_along_axis(lambda x: NumericalPython.convolve(x, NumericalPython.ones(size) / size, mode='valid'), axis=1, arr=values)
            scaler = MinMaxScaler()
            values = scaler.fit_transform(values)
            if x is None:
                x = values.reshape(1, values.shape[0], values.shape[1] - size)
            else:
                x = NumericalPython.concatenate((x, values.reshape(1, values.shape[0], values.shape[1] - size)))
        x = NumericalPython.transpose(x, (1, 2, 0))
        return x

    x = preprocess_data(Dictionary_data_values, window_size)
    y = dataframe_target['stable'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    
    # Build the model
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, iNumericalPythonut_shape=(x.shape[1], x.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
   
    model2 = Sequential()
    model2.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(x.shape[1], x.shape[2])))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(Flatten())
    model2.add(Dense(64, activation='relu'))
    model2.add(Dropout(dropout_rate))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Train the CNN model
    history2 = model2.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[early_stopping])

    # Train the model
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[early_stopping])
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss vs. Validation Loss')
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(history2.history2['loss'], label='Training Loss')
    plt.plot(history2.history2['val_loss'], label='Validation Loss')
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
    # Return the trained model and history
    return model, history


# Create a Flask web app


    




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
