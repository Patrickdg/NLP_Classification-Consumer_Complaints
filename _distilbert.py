#LIBRARIES
import processing as p
import model_funcs as mf

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense

from transformers import AutoTokenizer,TFBertModel
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

def model(train_x, test_x, train_y, test_y, model_name, max_length):
    n_classes = train_y.shape[1]
    #TOKENIZATION
    train_x, test_x, train_input_ids, train_attention_mask = mf.bert_tokenize(model_name, 
                                                                            AutoTokenizer, 
                                                                            TFDistilBertForSequenceClassification, 
                                                                            train_x, 
                                                                            test_x, 
                                                                            max_length)

    #ARCHITECTURE
    model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    embeddings = model(train_input_ids, attention_mask = input_mask)[0] 
    out = Dense(max_length, activation='relu')(embeddings)
    out = tf.keras.layers.Dropout(0.1)(out)
    out = Dense(32, activation = 'relu')(out)
    y = Dense(n_classes, activation = 'sigmoid')(out)

    model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
    model.layers[2].trainable = True

    #OPTIMIZER + COMPILATION
    optimizer = Adam(
        learning_rate=5e-05,
        epsilon=1e-08,
        decay=0.01,
        clipnorm=1.0)
    model.compile(
        optimizer = optimizer,
        loss = 'categorical_crossentropy', 
        metrics = ['acc'])

    return model, train_x, test_x

def train_model(model): 
    earlystopping = EarlyStopping(monitor = "val_loss", 
                                mode = "min", patience = 10, 
                                restore_best_weights = True)
    history = model.fit(
        x = {'input_ids': train_x['input_ids'],
            'attention_mask': train_x['attention_mask']},
        y = train_y,
        validation_data = (
            {'input_ids': test_x['input_ids'],
            'attention_mask': test_x['attention_mask']}, 
            test_y),
        epochs = 50,
        batch_size = 16,
        callbacks = [earlystopping])
    return history

if __name__ == "__main__":
    model_name = "distilbert-base-uncased"
    max_length = 50 #max token length of samples 
    
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    train_x, test_x, train_y, test_y = mf.bert_processing(train_x, test_x, train_y, test_y)
    bert, train_x, test_x = model(train_x, test_x, train_y, test_y, model_name, max_length)
    history = train_model(bert)
    accuracy, WAF1 = mf.bert_plot(model, history, test_x, test_y, label_map)
    mf.store_results(model_name, '', model, accuracy, WAF1, tf_model = 1)