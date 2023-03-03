# ML-Image-Classification-Flask-API
Flask API that can be used to make image predictions in batches of data scheduled overnight.

Please note, in order for these programs to work, all of the files and locations has to be exactly the same as is currently seen.

The files and purposes are as follows:
G1 & G2 are both folders containing 100 test images each. These are used to send to the API in order to obtain predictions.

The folder named 'fashion' contains the ML model that is used to make predictions.

'Model code setup' is the jupyter notebook as well as the data that was used to create the ML model. The data used here is the MNIST fashion dataset.

CNN Fashion project flow, is a flow chart that shows how the program works, with the input along with the output files.

Events-log is where all of the errors are recorded, such as missing input image folders or when the connection to the API was unable to be obtained.

'Model Performance.csv' saves all of the performance parameters of the model, such as the time it took to complete request, along with the accuracy measures.

Prediction API, is the API system that contains the ML model. The api is able to receive batches of data, process it and return the prediction results. It is run on a 'localhost'.

Request.py is the program that sends the request to the API. It takes in raw images, contained here within the G1/G1 folders, transforms it into integer data and then sends the data to the API in order to obtain the predictions.

Results.csv is where our prediction results are stored along with the certainty of that prediction.

requirements, is as always, just the necessary libraries and versions needed to run the applications.

Presentation explanation is a powerpoint of how the app works along with how some of the challenges was solved.
