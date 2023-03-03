import requests 
from datetime import datetime
from retry import retry 
import glob
import cv2
import numpy as np
import pandas as pd
 
@retry((ConnectionError, FileNotFoundError), tries=3, delay=60)
def data_upload():

    # Get correct Image file to upload.
    # Otherwise return a FileNotFound Error.
    try:
        # Set File Directory for images.
        col_dir = 'G2/*.png'

        # Create empty array that will hold all of our image's pixel data.
        final_array = np.zeros((1,784), int)

        # This loop goes through our file and retrieves all of our 'png' images.
        # It then changes these images into an array containing the pixel data.
        for img in glob.glob(col_dir):
            # Read in the image
            cv_img = cv2.imread(img, 0)
            # Resize the image
            cv_img = cv2.resize(cv_img, (28,28))
            # Reshape image into correct array size
            final = cv_img.reshape(28,28,1)
            # Flatten into a single array
            final = final.flatten()
            # Reshape array into the correct size.
            final = final.reshape(1,784)
            # Append each image's pixel array to our final array.
            final_array = np.append(final_array, final, axis=0)
        
        # Transform our array into a DataFrame and apply basic transformation.
        final_df = pd.DataFrame(final_array, dtype=int)
        final_df = final_df.drop(0)
       
    except:   
        # This except clause is when a file not found error will arise.
        # We then write this error to our error log in order to understand why things do not work.
        date = datetime.now()
        with open('Events-Log.txt','a') as event:
            event.write(f"{date} - File Not Found. No data could be sent. Process stoped. \n")
        raise FileNotFoundError


    # Send POST request to the API
    # If API server is not found, return ConnectionError.
    try:
        # URL used to upload data to our API
        url_predict = "http://127.0.0.1:5000/predict"

        # Additional POST information requirements.
        headers = {'Content-type': 'application/json'}

        # POST request to API.
        requests.request('POST', url_predict, data=final_df.to_json(),  headers=headers)

    except:
        # This except is when the API might not be up and running, we then write it to the Events-Log.

        date = datetime.now()
        with open('Events-Log.txt','a') as event:
            event.write(f"{date} - Connection could not be made. No POST request was sent to '/predict'. Process stoped. \n")
        raise ConnectionError

    
    # Return output from API function
    return 'DONE!'


data_upload()
