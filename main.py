import pymongo
import pandas as pd
from pymongo import MongoClient
from pandas import json_normalize

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from functions import prep_churn_train, prep_churn_new, split_churn_train, split_churn_new, training_random_forest, prediction_random_forest
import Mongo_fun as mon


def main():


        ### Load churn_train

        churn_train_orig = pd.read_csv('churn.csv')



        ### Load churn_new

        
        
        churn_new_orig = mon.load_churn_new()



        ## adjusting churn_new_orig

        def extract_first_element(value):
            if isinstance(value, list) and len(value) > 0:
                return value[0] 
            return value  


        churn_new_orig["PaperlessBilling"] = churn_new_orig["PaperlessBilling"].apply(extract_first_element)

        churn_new_orig["PaperlessBilling"] = churn_new_orig["PaperlessBilling"].replace({"YES": True, "NO": False})



        T_Services = churn_new_orig['Services'].apply(pd.Series)
        churn_new_orig = churn_new_orig.join(T_Services)


        ## Using the FUNCTIONS

        churn_train = prep_churn_train(churn_train_orig)


        churn_new = prep_churn_new(churn_new_orig)


        ### split_churn_train

        x_train,y_train,csi_train = split_churn_train(churn_train)


        ### split_churn_new

        x_new, csi_new = split_churn_new(churn_new)



        ### training_random_forest

        model = training_random_forest(100,3,1,x_train,y_train)



        ### predict_random_forest

        x_new = x_new.reindex(columns=x_train.columns, fill_value=0)


        churn_new_orig_with_predict = prediction_random_forest(model,x_new,churn_new_orig)



        ## Export to file

        churn_new_orig_with_predict.to_csv('churn_final_tab.csv', index=False)


        ## Accuracy

        y_train_pred_RandomForest = model.predict(x_train) # making a prediction based on "x_churn_new_without_csi" data features
        
        # Evaluation for Decision Tree
        test_acc = accuracy_score(y_train, y_train_pred_RandomForest)
        test_acc

        

if __name__ == '__main__':
     main()