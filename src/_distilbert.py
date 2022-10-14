"""LIBRARIES ======================================================"""
from telnetlib import SE
import model_funcs as mf

import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense

from transformers import AutoTokenizer, TFDistilBertForSequenceClassification

"""ARCHITECTURE ======================================================"""
def model(clf, hidden_layers = 0, hidden_units = 1024): 
    #Bert
    clf_model = clf.from_pretrained(MODEL_NAME)
    input_ids = Input(shape=(MAX_LEN,), name='input_ids', dtype='int32')
    input_mask = Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')
    embeddings = clf_model(input_ids, attention_mask = input_mask)[0]
    #Layers
    out = Dense(hidden_units, activation='relu')(embeddings)
    for h in range(hidden_layers): 
        out = Dense(hidden_units, activation='relu')(out)
    # out = tf.keras.layers.Dropout(0.1)(out)
    # out = Dense(32, activation = 'relu')(out)
    y = Dense(NUM_CLASSES, activation = 'softmax', name='outputs')(out)

    clf_model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    clf_model.layers[2].trainable = True

    clf_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRIC)
    return clf_model

"""EXECUTION ======================================================"""
if __name__ == "__main__":
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    
    #GENERAL PARAMS
    NUM_CLASSES = len(label_map)
    MODEL_SAVE_PATH = 'models/distilbert.h5'
    MODEL_NAME = "distilbert-base-uncased"
    BERT = TFDistilBertForSequenceClassification
    TOKENIZER = AutoTokenizer

    #TUNING PARAMS
    MAX_LEN = 150 #max token length of samples 

    NUM_ADD_HIDDEN_LAYERS = 2
    NUM_HIDDEN_UNITS = 100 #per hidden layer
    EPOCHS = 100
    BATCH_SIZE = 16
    PATIENCE = 5
    MIN_DELTA = 0.005

    LR = 5e-05
    EPSILON = 1e-08
    DECAY = 0.01
    CLIP_NORM = 1.0
    LOSS = 'categorical_crossentropy'
    METRIC = ['acc']
    OPTIMIZER = Adam(learning_rate=LR, 
                    epsilon=EPSILON, 
                    decay=DECAY, 
                    clipnorm=CLIP_NORM)
    CALLBACKS = EarlyStopping(monitor='val_acc',
                            min_delta=MIN_DELTA,
                            patience=PATIENCE, 
                            restore_best_weights=True)
    PARAMS = {var: eval(var) for var in ['MAX_LEN',
                                        'NUM_ADD_HIDDEN_LAYERS',
                                        'BATCH_SIZE',
                                        'EPOCHS',
                                        'PATIENCE',
                                        'LR', 
                                        'EPSILON',
                                        'DECAY',
                                        'CLIP_NORM'
                                        ]}

    #DATA PROCESSING 
    train_data = mf.bert_processing(TOKENIZER, MODEL_NAME,train_x, train_y, MAX_LEN)
    test_data = mf.bert_processing(TOKENIZER, MODEL_NAME, test_x, test_y, MAX_LEN)
    #COMPILATION, TRAINING
    MODEL = model(BERT, NUM_ADD_HIDDEN_LAYERS, NUM_HIDDEN_UNITS)
    MODEL, history = mf.bert_train(MODEL, train_data, test_data, EPOCHS, BATCH_SIZE, CALLBACKS)
    #EVALUATION
    y_preds, accuracy, f1 = mf.get_scores(MODEL, test_data, one_hot_encoded = 1)
    clf_report, conf_mat = mf.get_reports(test_data[1], y_preds, label_map)
    #STORAGE
    mf.store_results(MODEL_NAME, '', MODEL, PARAMS, accuracy, f1, clf_report, conf_mat, tf_model = 1)
    mf.tf_plot(history, MODEL_NAME, show = 1, save = 1)
