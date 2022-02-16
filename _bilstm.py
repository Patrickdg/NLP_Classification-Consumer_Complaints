import model_funcs as mf

import os
import pandas as pd
import numpy as np
import pickle

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, Flatten, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import schedules, SGD

def bilstm_processing(train_x, test_x, max_words, max_len): 
    train_x, tokenizer = mf.lstm_build_sequences(train_x, max_words, max_len)
    test_x, tokenizer = mf.lstm_build_sequences(test_x, max_words, max_len, tokenizer=tokenizer)
    word_index = tokenizer.word_index

    return train_x, test_x, word_index

"""ARCHITECTURE"""
def bilstm_model(embedding_dim, embedding_mat, max_words, max_len, num_classes): 
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    # model.add(Bidirectional(LSTM(embedding_dim, activity_regularizer=l2(0.001),  kernel_regularizer=l2(0.001)))) #, kernel_regularizer=l2(0.01)
    # model.add(Bidirectional(LSTM(embedding_dim, return_sequences=True))) #, kernel_regularizer=l2(0.01)
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()

    model.layers[0].set_weights([embedding_mat])
    model.layers[0].trainable = False

    return model

"""TRAINING"""
def bilstm_train(model, train_x, train_y, epochs, batch):
    #DECAY
    # optimizer = 'rmsprop'
    optimizer = SGD(learning_rate=1e-3, momentum=0.9, decay=1e-3/epochs)
    #COMPILATION
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
    earlystopping = EarlyStopping(monitor = "val_loss", 
                                    mode = "min", patience = 10, 
                                    restore_best_weights = True)
    history = model.fit(train_x, np.array(train_y),
                        epochs=int(epochs),
                        batch_size=int(batch),
                        validation_split=0.1, 
                        callbacks=[earlystopping])
    return history

def get_reports(test_y, y_preds, label_map):
    target_names = [k for k,v in label_map.items()]
    clf_report = classification_report(test_y, y_preds, target_names = target_names, output_dict = True)

    conf_mat = confusion_matrix(test_y, y_preds)
    return clf_report, conf_mat

if __name__ == "__main__": 
    OVERWRITE_FLAGS = {
        'word2vec': False, 
        'glove': True, 
        'fasttext': False,
        'custom': False
    }
    MODEL = 'bilstm'
    max_len = 300 # maximum length of input vectors
    max_words = 7500 # top n words in vocab
    EPOCHS = 100
    BATCH = 32
    params = {var: eval(var) for var in ['max_len', 'max_words', 'EPOCHS', 'BATCH']}

    #Data - raw
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    train_x, test_x, word_index = bilstm_processing(train_x, test_x, max_words, max_len)
    num_classes = len(list(label_map.keys()))
    
    #Run model
    for emb_name, emb_func in mf.embedding_funcs.items(): 
        if OVERWRITE_FLAGS[emb_name]: 
            model_name = f'{MODEL}-{emb_name}'
            model_path = f'models/{model_name}'
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            print(f"RUNNING {MODEL.upper()} - {emb_name} embeddings")
            #Get embeddings
            emb_dict = mf.embedding_funcs
            emb_func = emb_dict[emb_name]; embeddings_index = emb_func()
            embedding_dim, embedding_mat = mf.lstm_build_embed_mat(emb_name, embeddings_index, word_index, max_words)
            #Build, train model
            model = bilstm_model(embedding_dim, embedding_mat, max_words, max_len, num_classes)
            history = bilstm_train(model, train_x, train_y, EPOCHS, BATCH)
            y_preds, accuracy, WAF1 = mf.lstm_plot(model_name, model, history, test_x, test_y)
            clf_report, conf_mat = get_reports(test_y, y_preds, label_map)
            mf.store_results(MODEL, emb_name, model, params, accuracy, WAF1, clf_report, conf_mat, tf_model=1)
