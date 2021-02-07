import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot

from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from deep_fm_prunable import create_deepfm_functional_model, create_deepfwfm_functional_model, create_sparsity_schedule
from criteo_loader import CriteoLoader
import tempfile
from tqdm import tqdm
from train_util import create_log_and_config_file, save_model

hyper_params = {
    'lr': 0.01,
    'batch_size': 64,
    'epochs': 20,
    'hidden_dims': [400, 400, 400],
    'layer_prunables': [False, False, False],
    'emb_dim': 32,
    'use_numeric': False,
    'l2': 0.03,
    'train_path':'criteo/sample_train_05.txt',
    'test_path': 'criteo/sample_test_01.txt',
    'model_name': 'deepfwfm_sparse',
    'emb_sparsity': 0.8,
    'deep_sparsity': 0.8,
    'weight_matrix_sparsity': 0.8

}

def _get_criteo_dataset():
    train = hyper_params['train_path']
    test =  hyper_params['test_path']
    
    loader = CriteoLoader(train, test, use_numeric=hyper_params['use_numeric'])
    loader.load_index_mapper()
    # loader.fit_index_mappers()
    n_categs = loader.get_categ_count()
    train_dataset = tf.data.Dataset.from_generator(
        loader.train_generator,
        output_types=(tf.int32, tf.float32, tf.int8)
    )
    test_dataset = tf.data.Dataset.from_generator(
        loader.test_generator,
        output_types=(tf.int32, tf.float32, tf.int8)
    )
    return train_dataset, test_dataset, n_categs


def train():
    print('configs:')
    print(hyper_params)
    train_dataset, val_dataset, n_categs = _get_criteo_dataset()

    if hyper_params['emb_sparsity'] > 0:
        emb_sparsity = create_sparsity_schedule(hyper_params['emb_sparsity'], 71672)
    if hyper_params['deep_sparsity'] > 0:
        deep_sparsity = create_sparsity_schedule(hyper_params['deep_sparsity'], 71672)
    if hyper_params['weight_matrix_sparsity'] > 0:
        weight_matrix_sparsity = create_sparsity_schedule(
            hyper_params['weight_matrix_sparsity'], 71672
        )    
    
    
    model_for_pruning = create_deepfwfm_functional_model(
        n_categs, hyper_params['hidden_dims'], 
        hyper_params['layer_prunables'] , hyper_params['emb_dim'], 
        prune_emb_sparsity=emb_sparsity, prune_deep_sparsity=deep_sparsity,
        prune_weight_matrix_sparsity=weight_matrix_sparsity,
        l2=hyper_params['l2']
    )
    ###
    # build model
    model_for_pruning.build((hyper_params['batch_size'], len(n_categs)))
    ###
    print(model_for_pruning.summary())
    criteon = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate=hyper_params['lr'])
    # train_mse = tf.keras.metrics.MeanSquaredError()
    # val_mse = tf.keras.metrics.MeanSquaredError()
    train_auc = tf.keras.metrics.AUC()
    val_auc = tf.keras.metrics.AUC()
    train_bc = tf.keras.metrics.BinaryCrossentropy()
    val_bc = tf.keras.metrics.BinaryCrossentropy()


    csvlogger = create_log_and_config_file(hyper_params['model_name'], params=hyper_params)
    csvlogger.set_model(model_for_pruning)
    csvlogger.on_train_begin()
    unused_arg = -1

    model_for_pruning.optimizer = optimizer
    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model_for_pruning)
    log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir='pruning.log') # Log sparsity and other metrics in Tensorboard.
    log_callback.set_model(model_for_pruning)
    step_callback.on_train_begin()

    for epoch in range(hyper_params['epochs']):
        log_callback.on_epoch_begin(epoch=unused_arg) # run pruning callback
        csvlogger.on_epoch_begin(epoch=epoch)
        train_auc.reset_states()
        val_auc.reset_states()
        train_bc.reset_states()
        val_bc.reset_states()

        for index, coef, y in tqdm(train_dataset.shuffle(5000).batch(hyper_params['batch_size'])): 
            step_callback.on_train_batch_begin(batch=unused_arg) # run pruning callback
            with tf.GradientTape() as tape:
                logits = model_for_pruning([index, coef], training=True)
                loss = criteon(y, logits)  
            grads = tape.gradient(loss, model_for_pruning.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_for_pruning.trainable_variables))
            train_auc.update_state(y, logits)
            train_bc.update_state(y, logits)
            
            
        step_callback.on_epoch_end(batch=unused_arg) # run pruning callback
        for index, coef, y in val_dataset.batch(256): 
            logits = model_for_pruning([index, coef], training=False)
            loss = criteon(y, logits)
            val_auc.update_state(y, logits)
            val_bc.update_state(y, logits)        
                        
        print('+ epoch : ',epoch)
        print('    * train loss:', train_bc.result().numpy())
        print('    * val loss:', val_bc.result().numpy())
        print('    * train AUC:', train_auc.result().numpy())
        print('    * val AUC:', val_auc.result().numpy())
        step_callback.on_epoch_end(batch=unused_arg) # run pruning callback    
        csvlogger.on_epoch_end(epoch=epoch, logs={
            'train_loss': train_bc.result().numpy(),
            'val_loss': val_bc.result().numpy(),
            'train_auc': train_auc.result().numpy(),
            'val_auc': val_auc.result().numpy()            
        })
    csvlogger.on_train_end()
    save_model(model_for_pruning, 'deepfwfm_criteo/1')
                


if __name__ == "__main__":
    train()