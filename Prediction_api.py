import flask
from flask import request, jsonify
import numpy as np
import pandas as pd
from tensorflow import keras 
import time
from datetime import datetime

app = flask.Flask(__name__)

# Load the trained model
fashion_model = keras.models.load_model('fashion')

@app.route('/')
def home():
    return 'home'

@app.route('/predict', methods=['GET','POST'])
def predict():
    # Start of process
    process_start_time = time.perf_counter()

 
    # Get the JSON data
    json_data = request.get_json(force=True)

    # Read JSON data into pandas DataFrame
    df = pd.DataFrame.from_dict(json_data)

    # Convert the data to a numpy array
    data = np.array(df)

    # Reshape data into correct size for ML model
    data = data.reshape(data.shape[0], 28, 28, 1)
        
    # Normalize data for model
    data = data/255


    # Use the model to make predictions
    prediction_start_time = time.perf_counter()
    predictions = fashion_model.predict(data)
    prediction_end_time = time.perf_counter()


    # Now we start with transforming our predictions into a usefull CSV file. 
    pred_list = []
    col_names = ['Num','Item','Accuracy %']

    # Assignment of Indexes to Clothing item.
    clothing_groups = {0:'T-shirt/Top', 1:'Trouser', 2:'Pullover',3:'Dress',
         4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}

    """ Basic loop to run through all of our predictions, and append the, 
     1. Prediction image number , 
     2. Clothing item , 
     3. Prediction accuracy percentage"""
    for i in range(predictions.shape[0]):
        #Get highest predicted value percentage and it's index.
        max_val_percentage = round((predictions[i].max()) * 100  , 2)
        max_val_index = predictions[i].argmax()

        #Get the clothing item based on the index
        clothing_item = clothing_groups[max_val_index]

        # Append results to list
        pred_list.append([i , clothing_item , max_val_percentage])

    # Now we have our prediction list and can store them in a dataframe.
    result_df = pd.DataFrame(pred_list, columns=col_names)

    # Transform DataFrame into csv to return to client.
    result_df.to_csv('Results.csv',index=False)

    # End of process.
    process_end_time = time.perf_counter()



    """
    Our process has now ended, and the results have been returned to the user.
    We can now gather all of our Model Performance metrics.
    This is our Process / Prediction times.
    As well as our model Accuracy predcitions.
    """
    # Get all of the times
    total_process_time = round((process_end_time - process_start_time),3)
    # Time per prediction is total time for all predictions, divided by the total number of predictions.
    time_per_prediction = round(((prediction_end_time - prediction_start_time) / result_df.shape[0]),5)

    # Get the accuracy
    accuracy = result_df.groupby('Item')['Accuracy %'].mean().round(2)
    # Transform Groupby object into a DataFrame
    model_pm_current = accuracy.to_frame().T
    # Add all other relevant attributes.
    model_pm_current['Date'] = datetime.now()
    model_pm_current['Mean Accuracy'] = round((result_df['Accuracy %'].mean()),2)
    model_pm_current['Process Time'] = total_process_time
    model_pm_current['Time per Prediction'] = time_per_prediction

    # Load previous model data.
    model_pm = pd.read_csv('Model Performance.csv')
    # Add new data to existing dataset.
    new_model_pm = pd.concat([model_pm, model_pm_current])
    # Save our new Model performance data.
    new_model_pm.to_csv('Model Performance.csv', index=False)


    return jsonify(pred_list)
 
if __name__ == '__main__':
    # Start the Flask API
    app.run()

