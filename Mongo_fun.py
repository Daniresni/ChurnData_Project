import pymongo
import pandas as pd
from pymongo import MongoClient
from pandas import json_normalize

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

### Load churn_new

def load_churn_new():
    client = MongoClient('mongodb://localhost:27017')
    db = client['BDA']
    collection = db['customers']
    
    all_fields = set()
    for doc in collection.find({}, {"_id": 0}):
         all_fields.update(doc.keys())
    
    query = {field: {"$exists": True} for field in all_fields}
    documents =list(collection.find(query, {"_id": 0}))
    
    print(f"Number of documents: {len(documents)}")
  
    churn_new_orig = pd.DataFrame(documents)

    
    return churn_new_orig
