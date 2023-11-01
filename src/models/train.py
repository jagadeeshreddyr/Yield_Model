


import os
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor



def create_model(df:pd.DataFrame, loc:int, crop:str)-> None:

    """
    Creating Yield Prediction Model

    Parameters
    ----------

    df : DataFrame

        This df contains the training dataset

    loc : int 

        to loc the dataframe columns 

        
    crop: str

        to select the croptype

    Returns
    -------
        Create model and standard scalar model for crop mentioned 
        and save it has pickle file 

    """

    #select the dataframe based on crop type
    df = df.loc[train_df['Crop'] == crop]

    #input data loc will selct the columns based on index 3
    X = df.iloc[:,loc:-1].values

    #output data
    y = df.iloc[:,-1].values

    #standarize the input data
    scaler = StandardScaler()

    scaler.fit(X)

    X = scaler.fit_transform(X)


    ## Random forest model 
    random_model = RandomForestRegressor(n_estimators=100,random_state=0)


    random_model.fit(X,y)

    #pred = random_model.predict(X)

    #df['pred'] = pred
    
    # save the model in pickle format
    with open(f"../../model/{crop}/main.pkl", 'wb') as src:
        pickle.dump(random_model, src, protocol=3)

    with open(f"../../model/{crop}/slr.pkl","wb") as src:
        pickle.dump(scaler, src, protocol=3)




if __name__ == "__main__":

    '''
    To set the path to current directory
    '''

    dirname = os.path.dirname(os.path.abspath(__file__))
    os.chdir(dirname)

    # importing training csv after preprocessing
    train_df = pd.read_csv('../../data/processed/training.csv')


    # select any one crop and comment other crop
    # crop = 'Tomato'

    crop = 'Chili'

    #locate on column for input features 
    loc = 3

    create_model(train_df, loc, crop)




