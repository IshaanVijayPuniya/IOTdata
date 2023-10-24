
"""
DATA pipeline for IOT data at RENIVER of a  langZuaner smart Hydarulic press

PLEASE IMPLEMENT AN API ENDPOINT AND CALL THE TRAIN MODEL FUNCTION
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

downloads_folder = r"C:\Users\ishaa\Downloads\archive" 
file_list = [os.path.join(downloads_folder, filename) for filename in os.listdir(downloads_folder) if len(filename) < 8]
"""
output
Cooler condition : 3: close to total failure 20: reduced effifiency 100: full efficiency
stable flag: 0: conditions were unstable  1: stable conditions


"""
def create_lstm_model(x_train,lstm_units=64, dropout_rate=0.5):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
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
    num = 370
    plt.figure(figsize = (32,32))
    for i,title in enumerate(Dictionary_data_values.keys()): # enumerate i means 0 1 2 3 4 here
        
        plt.subplot(5,5,i+1)
        plt.title(f'{title}')
        for v in stables[:num]:
            plt.plot(Dictionary_data_values[title][v], color='red')
        plt.plot(Dictionary_data_values[title][stables[num]], label='stable', color='blue')
        for v in unstables[:num]:
            plt.plot(Dictionary_data_values[title][v], color = 'blue', alpha=0.5)
        plt.plot(Dictionary_data_values[title][unstables[num]], label='unstable', color ='red', alpha=0.7)
        plt.legend(loc='upper right')
        
    

    def preprocess_data(data_dict, size=1):
        x = None

        for key in data_dict:
            values = data_dict[key]
            if values.shape[1] == 6000:
                values = values[:, ::100] # STEPS of 100 to make 60 rows
            elif values.shape[1] == 600:
                values = values[:, ::10] # STEPS of 10 to make 60 rows
            assert values.shape[1] == 60
            """
        
            numpy.convolve(a, v, mode='full')[source]
                Returns the discrete, linear convolution of two one-dimensional sequences
                
            """
            values = NumericalPython.apply_along_axis(lambda x: NumericalPython.convolve(x, NumericalPython.ones(size) / size, mode='valid'), axis=1, arr=values)
            """
                            Ni, Nk = a.shape[:axis], a.shape[axis+1:]
                for ii in ndindex(Ni):
                    for kk in ndindex(Nk):
                        f = func1d(arr[ii + s_[:,] + kk])
                        Nj = f.shape
                        for jj in ndindex(Nj):
                            out[ii + jj + kk] = f[jj]
            """
            scaler = MinMaxScaler() # We can also use Standard scaler
            values = scaler.fit_transform(values)
            if x is None:
                x = values.reshape(1, values.shape[0],61 - size)  # some bug with shape[1]
            else:
                x = NumericalPython.concatenate((x, values.reshape(1, values.shape[0], 61 - size)))
                """
                numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")
                Join a sequence of arrays along an existing axis.
                """
        x = NumericalPython.transpose(x, (1, 2, 0))
        return x

    x = preprocess_data(Dictionary_data_values, window_size)
    y = dataframe_target['stable'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    
    # Build the model
   
    """
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
    """
    
    # Specify the hyperparameters and their possible values
    param_grid = {
        'lstm_units': [32, 64, 128],
        'dropout_rate': [0.3, 0.5, 0.7],
        'epochs': [50, 100, 150],
    }
    
    # Create GridSearchCV object
    grid_search = GridSearchCV(estimator=create_lstm_model(x_train), param_grid=param_grid, scoring=make_scorer(f1_score), cv=3)
    
    # Fit the GridSearchCV to your data
    grid_search.fit(x_train, y_train)
    
    # Get the best parameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    
    # Print the best parameters
    print("Best Parameters: ", best_params)

    # Train the best model with the best parameters
    history=best_model.fit(x_train, y_train, epochs=best_params['epochs'], validation_data=(x_test, y_test), callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)])

    # Evaluate the best model on training and test data
    train_loss, train_acc = best_model.evaluate(x_train, y_train)
    test_loss, test_acc = best_model.evaluate(x_test, y_test)

    print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}")
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    # Train the model
   
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss vs. Validation Loss')
    plt.show()
    plt.figure(figsize=(10, 5))
   

    # Evaluate the model on training and test data
    
    return best_model, history


# Create a Flask web app

build_train_model(downloads_folder,r"C:\Users\ishaa\Downloads\archive\profile.txt", window_size=5, lstm_units=64, dropout_rate=0.5, epochs=50, test_size=0.2)
    




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
