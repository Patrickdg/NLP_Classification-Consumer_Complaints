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
from sklearn.metrics import classification_report, confusion_matrix

from transformers import RobertaTokenizer, TFRobertaForSequenceClassification 

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

def model(bert_clf, x_train, train_y, model_name, max_length):
    #Setup 
    n_classes = train_y.shape[1]
    input_ids = x_train['input_ids']
    attention_mask = x_train['attention_mask']

    #ARCHITECTURE
    model = bert_clf.from_pretrained(model_name)
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    input_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
    embeddings = model(input_ids, attention_mask = input_mask)[0] 
    out = Dense(max_length, activation='relu')(embeddings)
    # out = tf.keras.layers.Dropout(0.1)(out)
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

    return model

def train_model(model, x_train, train_y, x_test, test_y, epochs, batch): 
    earlystopping = EarlyStopping(monitor = "val_acc", 
                                mode = "max", patience = 5, 
                                restore_best_weights = True)
    history = model.fit(
        x = {'input_ids': x_train['input_ids'],
            'attention_mask': x_train['attention_mask']},
        y = train_y,
        validation_data = (
            {'input_ids': x_test['input_ids'],
            'attention_mask': x_test['attention_mask']}, 
            test_y),
        epochs = int(epochs),
        batch_size = int(batch),
        callbacks = [earlystopping])
    return history

def get_reports(test_y, y_preds, label_map):
    target_names = [k for k,v in label_map.items()]

    y_preds_max = []
    for pred in y_preds: 
        y_preds_max.append(np.argmax(pred))
    test_max = []
    for pred in test_y: 
        test_max.append(np.argmax(pred))

    clf_report = classification_report(test_max, y_preds_max, target_names = target_names, output_dict = True)

    conf_mat = confusion_matrix(test_max, y_preds_max)
    return clf_report, conf_mat

if __name__ == "__main__":
    # import imp; imp.reload(mf)
    model_name = "roberta-base"
    bert_clf = TFRobertaForSequenceClassification
    max_length = 100 #max token length of samples 
    epochs = 50
    batch = 16
    params = {var: eval(var) for var in ['max_length', 'epochs', 'batch']}
    
    train_x, test_x, train_y, test_y, label_map  = mf.get_train_test()
    train_x, test_x, train_y, test_y = mf.bert_processing(train_x, test_x, train_y, test_y)
    x_train, x_test = mf.bert_tokenize(model_name, 
                                        RobertaTokenizer, 
                                        train_x,
                                        test_x,
                                        max_length)
    bert = model(bert_clf, x_train, train_y, model_name, max_length)
    history = train_model(bert, x_train, train_y, x_test, test_y, epochs, batch)
    y_preds, accuracy, F1 = mf.bert_plot(model_name, bert, history, x_test, test_y, label_map)
    clf_report, conf_mat = get_reports(test_y, y_preds, label_map)
    mf.store_results(model_name, '', bert, params, accuracy, F1, clf_report, conf_mat, tf_model = 1)
