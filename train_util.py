import os
import datetime
import json
import tensorflow as tf

def create_log_and_config_file(model_name, params): 
    'create log and config file for each training.'
    if not os.path.exists('logs'):
        os.mkdir('logs')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #save configs:
    config_name =  current_time + '{}_config.json'.format(model_name)
    with open(os.path.join('logs', config_name), 'wt') as f:
        json.dump(params, f, indent=4)
    train_log_name = current_time + '{}_log.csv'.format(model_name)
    csvlogger = tf.keras.callbacks.CSVLogger(os.path.join('logs', train_log_name))
    return csvlogger

def save_model(model, name):
    model_folder_path = os.path.join('modelfiles', name)
    path = ''
    for token in model_folder_path.split('/'):
        path = os.path.join(path, token)
        if not os.path.exists(path):
            os.mkdir(path)        
    tf.saved_model.save(model, model_folder_path)    