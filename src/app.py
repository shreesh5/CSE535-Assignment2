import csv
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import keras 
import warnings

warnings.filterwarnings('ignore')
app = Flask(__name__)


def convert_to_csv(json_data):

    # Converting to Pandas Dataframe
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    csv_data = np.zeros((len(json_data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(json_data[i]['score'])
        for obj in json_data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    pd.DataFrame(csv_data, columns=columns).to_csv('data.csv', index_label='Frames#')
    df = pd.read_csv('data.csv')[["leftShoulder_x", "leftShoulder_y", "rightShoulder_x", "rightShoulder_y", "leftElbow_x", "leftElbow_y", "rightElbow_x", "rightElbow_y", "leftWrist_x", "leftWrist_y", "rightWrist_x", "rightWrist_y"]]
    return df
    
def convert_to_np(csv_data):
    # Preparing input CSV Data

    # If num_rows is less than 232, impute mean rows
    if(len(csv_data) < 232):
        maxlen = 232
        dflen = len(csv_data)
        csv_data = csv_data.append([pd.Series()]*(maxlen-dflen), ignore_index=True)
        csv_data.fillna(csv_data.mean(), inplace=True)
    
    # Else trim rows equally from top and bottom
    elif(len(csv_data) > 232):
        minlen = 232
        dflen = len(csv_data)
        dellen = dflen-minlen
        startlen = int(dellen/2)
        endlen = dflen - (dellen - startlen)
        csv_data = csv_data.iloc[startlen:endlen].reset_index(drop=True)
    
    # 3D Input Array
    x = np.array(csv_data)

    # Reshaping to 2D Array
    x_input = x.reshape(1,-1)

    # Necessary Scaling For SVMs
    std_scaler = StandardScaler()
    std_scaler.fit(X_final)
    std_scaled_x_input = std_scaler.transform(x_input)

    # Necessary Scaling for NN, XGBoost
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(X_final)
    mm_scaled_x_input = mm_scaler.transform(x_input)
    return std_scaled_x_input, mm_scaled_x_input



@app.route('/')
def home_endpoint():
    return "You have reached Group 21's area. Please go to /predict !"


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
         # Get data posted as a json
        data = request.get_json()

        # Convert json to csv
        csv_data = convert_to_csv(data)

         # Convert csv to numpy array after selecting features
        std_np_data, mm_np_data = convert_to_np(csv_data)

        # 1. Predicting using SVM One Vs One
        pred1 = model1_svm_ovo.predict(std_np_data)
        
        # 2. Predicting using SVM One Vs Rest
        pred2 = model2_svm_ovr.predict(std_np_data)
        
        # 3. Predicting using XGBoost
        pred3 = model3_xgboost.predict(mm_np_data)
        
        # Session Handling
        session = keras.backend.get_session() # Maintaining Session
        init = tf.global_variables_initializer() # Global Variables to Tensorflow Memory
        session.run(init) # Initializing Session
        
        # Preparing Neural Network Model
        model4_nn = pickle.load(open('neural_network_model_new.pkl','rb'))
        graph = tf.get_default_graph()
        
        # 4. Predicting using Neural Network
        with graph.as_default():
            pred4 = model4_nn.predict(mm_np_data)
            pred4 = np.argmax(pred4,axis=1)

        tf.keras.backend.clear_session() # Clearing Session

        # Preparing JSON response for service
        #mapdict = {'communicate':0, 'hope':1, 'mother':2, 'really':3, 'fun':4, 'buy':5}
        maplist = ['communicate', 'hope', 'mother', 'really', 'fun', 'buy'] # Integer to Label Mapping
        predictions = {"1": maplist[int(pred1)], "2": maplist[int(pred2)], "3": maplist[int(pred3)], "4": maplist[int(pred4[0])] } # Output
        print(predictions)
    return jsonify(predictions)
    

if __name__ == '__main__':
    # Loading Training Data
    X_final = pickle.load(open('Xfile_final_for_scaling.pkl','rb'))
    
    # Preparing Trained Models for Prediction
    model1_svm_ovo = pickle.load(open('svm_ovo_model_rbf.pkl','rb'))
    model2_svm_ovr = pickle.load(open('svm_ovr_model_all.pkl','rb'))
    model3_xgboost = pickle.load(open('xgboost_ubuntu_model.pkl','rb'))

    # Run the service indefinitely
    app.run(host='0.0.0.0', port=5000)
