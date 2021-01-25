import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot
from tqdm import tqdm
from models.deep_fm_prunable import (
    create_deepfwfm_functional_model, create_sparsity_schedule
)
from ml100k_loader import ML100kLoader
from ml1m_loader import ML1MLoader
from train_util import create_log_and_config_file, save_model

hyper_params = {
    'lr': 0.001,
    'batch_size': 128,
    'epochs': 1,
    'hidden_dims': [400, 400, 400],
    'layer_prunables': [False, False, False],
    'emb_dim': 32,
    'l2': 0.1,
    'use_user_info':True,
    'use_movie_info': True,
    'data':'1m',
    'model_name': 'deepfwfm',
    'emb_sparsity': 0.9,
    'deep_sparsity': 0.9,
    'weight_matrix_sparsity': 0.9
}

def _get_ml_dataset(data='1m'):
    def _post_process(gen):
        for idx, y in gen:
            coef = [int(i != 0) for i in idx]
            yield idx, coef, float(y)/5
    use_user = hyper_params['use_user_info']
    use_movie = hyper_params['use_movie_info']
    if data == '1m':
        loader = ML1MLoader()
    else:
        loader = ML100kLoader()
    data = [x for x, y in loader.train_rating_generator(use_movie, use_user, False)]
    n_categs = [
        max([d[i] for d in data]) + 1 for i in range(len(data[0]))
    ]
    train_dataset = tf.data.Dataset.from_generator(
        lambda: _post_process(loader.train_rating_generator(use_movie, use_user, False)),
        output_types=(tf.int32, tf.float32, tf.float32),
    )
    val_dataset = tf.data.Dataset.from_generator(
        lambda: _post_process(loader.val_rating_generator(use_movie, use_user, False)),
        output_types=(tf.int32, tf.float32, tf.float32),
    )
    return train_dataset, val_dataset, n_categs


def train():
    print('configs:')
    print(hyper_params)
    train_dataset, val_dataset, n_categs = _get_ml_dataset(hyper_params['data'])
    emb_sparsity = None
    deep_sparsity = None
    weight_matrix_sparsity = None
    if hyper_params['emb_sparsity'] > 0:
        emb_sparsity = create_sparsity_schedule(hyper_params['emb_sparsity'], 13000)
    if hyper_params['deep_sparsity'] > 0:
        deep_sparsity = create_sparsity_schedule(hyper_params['deep_sparsity'], 13000)
    if hyper_params['weight_matrix_sparsity'] > 0:
        weight_matrix_sparsity = create_sparsity_schedule(
            hyper_params['weight_matrix_sparsity'], 13000
        )

    model_for_pruning = create_deepfwfm_functional_model(
        n_categs, [300, 300, 300], hyper_params['layer_prunables'],
        hyper_params['emb_dim'], hyper_params['l2'],
        prune_emb_sparsity=emb_sparsity, prune_deep_sparsity=deep_sparsity,
        prune_weight_matrix_sparsity=weight_matrix_sparsity
    )

    print(model_for_pruning.summary())
    criteon = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_params['lr'])
    train_mse = tf.keras.metrics.MeanSquaredError()
    val_mse = tf.keras.metrics.MeanSquaredError()
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
        csvlogger.on_epoch_begin(epoch=epoch)
        log_callback.on_epoch_begin(epoch=unused_arg)
        train_mse.reset_states()
        val_mse.reset_states()

        for index, coef, y in tqdm(train_dataset.shuffle(80000).batch(hyper_params['batch_size'])):
            step_callback.on_train_batch_begin(batch=unused_arg)
            with tf.GradientTape() as tape:
                logits = model_for_pruning([index, coef], training=True)
                loss = criteon(y, logits)
            grads = tape.gradient(loss, model_for_pruning.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_for_pruning.trainable_variables))
            train_mse.update_state(y, logits)

        step_callback.on_epoch_end(batch=unused_arg)
        for index, coef, y in val_dataset.batch(256):
            logits = model_for_pruning([index, coef], training=False)
            loss = criteon(y, logits)
            val_mse.update_state(y, logits)

        print('+ epoch : ', epoch)
        print('    * train loss:', train_mse.result().numpy() * 25)
        print('    * val loss:', val_mse.result().numpy() * 25)
        csvlogger.on_epoch_end(epoch=epoch, logs={
            'train_mse': train_mse.result().numpy() * 25,
            'val_mse': val_mse.result().numpy() * 25
        })
        step_callback.on_epoch_end(batch=unused_arg) # run pruning callback

    csvlogger.on_train_end()
    save_model(model_for_pruning, 'deepfwfm_ml1m/1')

if __name__ == "__main__":
    train()