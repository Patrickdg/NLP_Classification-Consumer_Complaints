import model_funcs as mf

import os
import pandas as pd
import numpy as np
import nltk
import pickle

import sklearn
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__": 
    MODEL = 'svm'
    grid_params = {
        'C': [0.1], 
        'kernel': ['rbf'],
        'gamma': [5]
    }
    #Data - raw
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    #Model
    pipeline = Pipeline([
        ("Normalize", Normalizer()),
        ("SVC", GridSearchCV(SVC(), grid_params))])
    #Embeddings
    for emb_name, emb_func in mf.embedding_funcs.items(): 
        if not os.path.exists(f'models/{MODEL}-{emb_name}.pkl'): 
            print(f"RUNNING {MODEL.upper()} - {emb_name} embeddings")
            train_vecs, test_vecs = mf.get_embedding_vecs(emb_name)
            model, y_preds, accuracy, weighted_f1 = mf.run_model(pipeline, train_vecs, train_y, test_vecs, test_y, label_map)
            mf.store_results(MODEL, emb_name, model, accuracy, weighted_f1)
