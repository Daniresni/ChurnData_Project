# Functions - Oz Reshef and Daniel Resnikow

import pymongo
import pandas as pd
from pymongo import MongoClient
from pandas import json_normalize

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


## Creating the FUNCTIONS

def prep_churn_train (df):
    df = df.rename(columns=str.lower)

    df.churn = df.churn == 'Yes'
    df = df.astype({"churn": 'int64'})


    df.dependents = df.dependents == 'Yes'
    df = df.astype({"dependents": 'int64'})
    
    df.partner = df.partner == 'Yes'
    df = df.astype({"partner": 'int64'})
    
    df.phoneservice = df.phoneservice == 'Yes'
    df = df.astype({"phoneservice": 'int64'})
    
    df.onlinesecurity = df.onlinesecurity == 'Yes'
    df = df.astype({"onlinesecurity": 'int64'})
    
    df.deviceprotection = df.deviceprotection == 'Yes'
    df = df.astype({"deviceprotection": 'int64'})
    
    df.techsupport = df.techsupport == 'Yes'
    df = df.astype({"techsupport": 'int64'})
    
    df.streamingtv = df.streamingtv == 'Yes'
    df = df.astype({"streamingtv": 'int64'})
    
    df.streamingmovies = df.streamingmovies == 'Yes'
    df = df.astype({"streamingmovies": 'int64'})
    
    df.paperlessbilling = df.paperlessbilling == 'Yes'
    df = df.astype({"paperlessbilling": 'int64'})

        
    df['multiplelines'] = df['multiplelines'].replace({
        'Yes': 1,
        'No': 0,
        'No phone service': 0
    })
    
    df['onlinebackup'] = df['onlinebackup'].replace({
        'Yes': 1,
        'No': 0,
        'No internet service': 0
    })


    #drop
    df = df.drop(['totalcharges'], axis = 1)

    customer_id = df['customerid']
    # get dumies
    df = pd.get_dummies(df)
    ##df['customerid'] = customer_id

    columns_to_convert = [
        'gender_Female', 'gender_Male',
        'internetservice_DSL', 'internetservice_Fiber optic',
        'internetservice_No', 'contract_Month-to-month', 'contract_One year',
        'contract_Two year', 'paymentmethod_Bank transfer (automatic)',
        'paymentmethod_Credit card (automatic)',
        'paymentmethod_Electronic check', 'paymentmethod_Mailed check',
            ]
    
    # Convert selected columns to boolean integers
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    df = df.astype(float) # Let's convert all data to float because some modules warn against other types
    df['customerid'] = customer_id
   # Check for nulls
    null = df.isna().sum().sum()
    print (f'There are {null} null values')
    
    # Check for DataType
    dtype = df.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
    
    print('')
    

    return df



# Prep_churn_new (df)

def prep_churn_new (df):
    df = df.rename(columns=str.lower)
  
    df.dependents = df.dependents == 'Yes'
    df = df.astype({"dependents": 'int64'})
    
    df.partner = df.partner == 'Yes'
    df = df.astype({"partner": 'int64'})
    
    df.phoneservice = df.phoneservice == 'Yes'
    df = df.astype({"phoneservice": 'int64'})
    
    df.onlinesecurity = df.onlinesecurity == 'Yes'
    df = df.astype({"onlinesecurity": 'int64'})
    
    df.deviceprotection = df.deviceprotection == 'Yes'
    df = df.astype({"deviceprotection": 'int64'})
    
    df.techsupport = df.techsupport == 'Yes'
    df = df.astype({"techsupport": 'int64'})
    
    df.streamingtv = df.streamingtv == 'Yes'
    df = df.astype({"streamingtv": 'int64'})
    
    df.streamingmovies = df.streamingmovies == 'Yes'
    df = df.astype({"streamingmovies": 'int64'})
    
    df.paperlessbilling = df.paperlessbilling == 'Yes'
    df = df.astype({"paperlessbilling": 'int64'})

    df.multiplelines = df.multiplelines == 'Yes'
    df = df.astype({"multiplelines": 'int64'})

    df.onlinebackup = df.onlinebackup == 'Yes'
    df = df.astype({"onlinebackup": 'int64'})
    
    #rename

    #drop
    df = df.drop(['totalcharges'], axis = 1)
    df = df.drop(['services'], axis = 1)

    

    customer_id = df['customerid']
    
    # get dumies
    df = pd.get_dummies(df)

    columns_to_convert = [
        'gender_Female', 'gender_Male', 'contract_Month-to-month', 'contract_One year',
        'contract_Two year', 'paymentmethod_Bank transfer (automatic)',
        'paymentmethod_Credit card (automatic)',
        'paymentmethod_Electronic check', 'paymentmethod_Mailed check',
            ]
    
    # Convert selected columns to boolean integers
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    df = df.astype(float) # Let's convert all data to float because some modules warn against other types

    df['customerid'] = customer_id
   # Check for nulls
    null = df.isna().sum().sum()
    print (f'There are {null} null values')
    
    # Check for DataType
    dtype = df.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
      
   
    
    print('')
    
    # Check for nulls
    null = df.isna().sum().sum()
    print (f'There are {null} null values')
    
    # Check for DataType
    dtype = df.dtypes.unique()[0]
    print (f'There are {dtype} dtype')
    
    
    print('')

    return df



### Fun - split_churn_train

def split_churn_train(df):

    label = 'churn'
    csi = 'customerid'

    x_train = df.drop(label, axis=1)
    x_train = x_train.drop(csi, axis=1)
    y_train = df[label]
    csi_train = df[csi]
    
    return x_train,y_train,csi_train


### Fun - split_churn_new

def split_churn_new(df):
    csi = 'customerid'

    x_new = df.drop(csi, axis=1)
    csi_new = df[csi]
    
    return x_new, csi_new


### Fun - training_churn_forest

def training_random_forest(n,m,r,x_train,y_train):

    model = RandomForestClassifier(n_estimators=n, max_depth=m, random_state=r)
    model.fit(x_train, y_train)
    
    return model


### Fun - prediction_random_forest

def prediction_random_forest(model,x_new,df_orig):

    y_new = model.predict(x_new) 
    y_new = pd.Series(y_new,name='predict')
    output = df_orig.join(y_new)
    
    return output


### Fun - random_forest_feature_importance

def random_forest_feature_importance(model,x_new):

    feature_importances = model.feature_importances_ # applying the method "feature_importances_" on the algorithm
    features = x_new.columns # all the features
    stats = pd.DataFrame({'feature':features, 'importance':feature_importances}) # creating the data frame
    print(stats.sort_values('importance', ascending=False)) # Sorting the data frame

    stats_sort = stats.sort_values('importance', ascending=True)
    stats_sort.plot(y='importance', x='feature', kind='barh')
    plt.title('Feature Importance of Random Forest')
    plt.show()


