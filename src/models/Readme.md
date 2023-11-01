


1 - src/models/data_processing.py: 

    This file is responsible for preprocessing raw sensor data. It likely performs data cleaning, normalization, and feature engineering to prepare the data for training the models. The processed training data is saved in a CSV file located at data/processed/training.csv.

2 - src/models/visualize_data.py: 
    
    This file handles the visualization of the sensor data. It probably creates plots and visualizations to gain insights into the sensor data and its patterns. The visualizations are saved in the directory data/output/visual/sensor/.

3 - src/models/train.py: 

    This file is used for training models specific to tomato and chili crops. It likely implements machine learning or deep learning algorithms for training the crop-specific models. Once trained, the models are saved in the directory model/crop/.

4 - src/models/predict.py: 

    This file is responsible for importing the trained models and using them to make predictions. It takes the preprocessed sensor data as input and outputs predictions based on the trained models.


5 - src/models/sensor_data.py: 

    This file contains the implementation of the deep learning model method. It seems to be the first approach for creating models and making predictions using deep learning techniques for the sensor data.
