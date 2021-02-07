import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow_model_optimization as tfmot
from itertools import chain

class Embedding_Prunalbe(keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    def __init__(self, dropout_rate, input_dim ,output_dim, l2=0):
        super().__init__()
        self.dropout = keras.layers.Dropout(dropout_rate)
        if l2>0:
            self.embedding = keras.layers.Embedding(
                input_dim, output_dim, 
                embeddings_regularizer=keras.regularizers.L2(l2=l2)
            )
        else:
            self.embedding = keras.layers.Embedding(input_dim, output_dim)
        
    def get_config(self,):
        return {
            'dropout_config' : self.dropout.get_config(),
            'embedding_config' : self.embedding.get_config()
        }
    
    def get_embedding_weights(self,): #-> List
        return self.embedding.weights
    
    def build(self, input_shape):
        super().build(input_shape)
        self.dropout.build(input_shape)
        self.embedding.build(input_shape)
        
    def get_prunable_weights(self):
        return self.embedding.weights
    
    # @tf.function(input_signature=[
    #     tf.TensorSpec([None,], tf.int32), 
    # ]) 
    def call(self, index):
        return self.dropout(self.embedding(index))


class Deep_Prunable(keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer):
    'deep part of deepFM'
    def __init__(self,
                 hidden_dims:list,
                 layer_prunables:list,
                 batch_norm=True):
        '''        
        layer_prunables: List[bool]: A list with li[i] imply the ith dense layer is prunable or not.
        '''
        assert type(hidden_dims) is list and len(hidden_dims)>0, \
               "args hidden_dim should be a list with at least one int value"               
        super().__init__()     
        self.layer_prunables = layer_prunables
        self.hidden_dims = hidden_dims   
        #self.sec_embs = sec_embs
        # self.deep = keras.models.Sequential()
        # #first fc:
        # self.deep.add(keras.layers.Dense(
        #     hidden_dims[0], input_shape=(input_shape,), activation='relu'
        # ))    
        # #other fc:
        # for dim in hidden_dims[1:]:
        #     self.deep.add(keras.layers.Dense(dim, activation='relu'))
        # #output layer:
        # self.deep.add(keras.layers.Dense(1, activation='relu'))
        self.denses = []
        self.batch_norms = []
        self._create_dense()
        self._create_batch_norm()
        

    def _create_dense(self,):
        #first layer:    
        for dim in self.hidden_dims:
            self.denses.append(keras.layers.Dense(dim, activation='relu'))
        # last layer
        self.denses.append(keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))
    
    def _create_batch_norm(self,):
        for _ in range(len(self.denses)-1):
            self.batch_norms.append(keras.layers.BatchNormalization())

    def get_config(self,):
        return {
            'layers': {
                l.name : l.get_config() for l in self.denses
            },
            'layer_prunables': self.layer_prunables,
            'hidden_dims': self.hidden_dims,
        }

    def build(self, input_shape):        
        super().build(input_shape)
        for dense in self.denses:
            dense.build(input_shape)
            input_shape = (dense.units,)

    def get_prunable_weights(self):        
        # return list(chain(*[
        #     layer.weights for prunable, layer in zip(self.layer_prunables ,self.denses) 
        #     if prunable
        # ]))
        return [
            layer.kernel for prunable, layer in zip(self.layer_prunables ,self.denses) 
            if prunable
        ]

    # @tf.function(input_signature=[tf.TensorSpec([None, None,], tf.float32)])
    def call(self, sec_coef_v):
        res = sec_coef_v        
        for dense, bn in zip(self.denses[:-1], self.batch_norms):
            res = dense(res)
            res = bn(res)
        res = self.denses[-1](res)
        return res

def create_sequential_deep_model(hidden_dims:list, layer_prunables:list, sparsity_schedules, batch_norm=True):
    '''
    hidden_dims: List[int].
    layer_prunables: List[Bool]. whether to prune n-th dense layer.
    sparsity_schedules: List or tfmot.sparsity.keras.ConstantSparsity. The sparsity shedule.
    '''
    assert len(hidden_dims) == len(layer_prunables), \
           'length of hidden_dims and layer_prunables should be equal'
    if not isinstance(sparsity_schedules, list):
        sparsity_schedules = [sparsity_schedules] * len(hidden_dims)
    model = tf.keras.Sequential([])
    for dim, prune, sparse in zip(hidden_dims, layer_prunables, sparsity_schedules):
        layer = tf.keras.layers.Dense(dim)
        if prune:
            layer = tfmot.sparsity.keras.prune_low_magnitude(layer, sparse)
        model.add(layer)
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
    # add final output layer:
    model.add(tf.keras.layers.Dense(1))
    return model
        


