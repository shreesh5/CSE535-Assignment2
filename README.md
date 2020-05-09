# CSE535-Assignment2

1. Create a files for data and labels in one of 2 ways.  
	(a) Use the shared link of the data which was provided and run the MC_Assignment_2_Data Preparation_Google_Collab.ipynb file
	
	OR
	
	(b) Create a Jupyter notebook in the CSV file (folder in which all the different sign folders are) and run the MC_Assignment_2_Data Preparation_Local.ipynb file
2. Once the X_final.pkl and Y_final.pkl files have been created during Step 1, run the notebooks for each Machine Learning Model.  
	(a) You can upload the X_final.pkl and Y_final.pkl files to your Google Drive and run the Jupyter Notebooks for the Neural Network and the XGBoost Decision Tree  
	(b) You can copy the Jupyter Notebooks for both the SVMs to the location of the X_final.pkl and Y_final.pkl file and run them  
3. After step 2, you would have created a .pkl file for each model. You can add these files to the same location as the app.py file.  
4. After running the Data Preparation file (either one), a Xfile_final_for_scaling.pkl would have been created. This is needed in order to scale the data for the new input while testing our service. Copy this file to the same location as the app.py file.  
5. Before running the app.py file, you must change the ip address mentioned from '0.0.0.0' to '127.0.0.1' so that it can run on a localhost.
