import model_funcs as mf

import os
import pandas as pd
import numpy as np
import pickle
import gensim

import xgboost as xgb
from xgboost import XGBClassifier
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

if __name__ == "__main__": 
    MODEL = 'xgb'
    grid_params = {
        'gamma': [0.5],
        'max_depth': [3],
        'sampling_method': ['gradient_based']
        }
    #Data - raw
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    #Model
    pipeline = Pipeline([
        ("XGB", GridSearchCV(XGBClassifier(use_label_encoder = False,
                                            eval_metric = 'mlogloss'),
                                            grid_params))])
    #Embeddings
    for emb_name, emb_func in mf.embedding_funcs.items(): 
        if os.path.exists(f'models/{MODEL}-{emb_name}.pkl'):
            continue 
        print(f"RUNNING {MODEL.upper()} - {emb_name} embeddings")
        train_vecs, test_vecs = mf.get_embedding_vecs(emb_name)
        model, y_preds, accuracy, weighted_f1 = mf.run_model(pipeline, train_vecs, train_y, test_vecs, test_y, label_map)
        mf.store_results(MODEL, emb_name, model, accuracy, weighted_f1)
