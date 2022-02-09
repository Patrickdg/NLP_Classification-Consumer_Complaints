import model_funcs as mf

import os
import pandas as pd
import numpy as np
import pickle

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, Flatten
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import Callback, EarlyStopping

def bilstm_processing(train_x, test_x, max_words, max_len): 
    train_x, tokenizer = mf.lstm_build_sequences(train_x, max_words, max_len)
    test_x, tokenizer = mf.lstm_build_sequences(test_x, max_words, max_len, tokenizer=tokenizer)
    word_index = tokenizer.word_index

    return train_x, test_x, word_index

def bilstm_model(embedding_dim, embedding_mat, max_words, max_len, num_classes): 
    #Architecture
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(embedding_dim)))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.summary()

    model.layers[0].set_weights([embedding_mat])
    model.layers[0].trainable = False

    return model

def bilstm_train(model, train_x, train_y):
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
    earlystopping = EarlyStopping(monitor = "val_loss", 
                                    mode = "min", patience = 10, 
                                    restore_best_weights = True)
    history = model.fit(train_x, np.array(train_y),
                        epochs=25,
                        batch_size=32,
                        validation_split=0.1, 
                        callbacks=[earlystopping])

    return history

if __name__ == "__main__": 
    MODEL = 'bilstm'
    max_len = 300 # maximum length of input vectors
    max_words = 10000 # top n words in vocab
    # embedding_dim = 300
    #Data - raw
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    train_x, test_x, word_index = bilstm_processing(train_x, test_x, max_words, max_len)
    num_classes = len(list(label_map.keys()))
    #Embeddings
    for emb_name, emb_func in mf.embedding_funcs.items(): 
        if os.path.exists(f'models/{MODEL}-{emb_name}.pkl'):
            continue 
        if os.path.exists(f'models/{MODEL}-{emb_name}/saved_model.pb'):
            continue 
        print(f"RUNNING {MODEL.upper()} - {emb_name} embeddings")
        emb_dict = mf.embedding_funcs
        emb_func = emb_dict[emb_name]; embeddings_index = emb_func()
        embedding_dim, embedding_mat = mf.lstm_build_embed_mat(emb_name, embeddings_index, word_index, max_words)
        
        model = bilstm_model(embedding_dim, embedding_mat, max_words, max_len, num_classes)
        history = bilstm_train(model, train_x, train_y)
        accuracy, WAF1 = mf.lstm_plot(model, history, test_x, test_y)

        mf.store_results(MODEL, emb_name, model, accuracy, WAF1, tf_model=1)
