import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude
from .prunable_layers import Deep_Prunable, Embedding_Prunalbe
from .fm_layers import FactorizationMachines, FieldWeightedFactorizationMachine


def create_sparsity_schedule(target_sparsity, start, end=-1, freq=300, mode='constant'):
    'currently only support mode: constant'
    if mode == 'constant':
        return tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity, start, end, freq
        )

def create_deepfwfm_functional_model(n_categs, hidden_dims, layer_prunables, emb_dim, 
                                     l2=0.05, prune_emb_sparsity=None, prune_deep_sparsity=None,
                                     prune_weight_matrix_sparsity=None ):
    index_inp = keras.layers.Input(shape=(None,), dtype=tf.int32, name='index')
    coef_inp =  keras.layers.Input(shape=(None,), dtype=tf.float32, name='coef')
    num_field = len(n_categs)
    #embeddings:
    second_embeddings = []
    l2_reg = keras.regularizers.L2(l2=l2)

    for n_categ in n_categs:
        sec_emb = tf.keras.layers.Embedding(
            n_categ, emb_dim, embeddings_regularizer=l2_reg
        )        
        if prune_emb_sparsity:
            sec_emb = prune_low_magnitude(sec_emb, prune_emb_sparsity)
        second_embeddings.append(sec_emb) 
    #fwfm
    weighted_matrix_dense = tf.keras.layers.Dense(num_field, use_bias=False, kernel_regularizer=l2_reg)
    field_embedding = tf.keras.layers.Embedding(num_field, emb_dim, embeddings_regularizer=l2_reg)
    if prune_emb_sparsity:
        field_embedding = prune_low_magnitude(field_embedding, prune_emb_sparsity)
        
    if prune_weight_matrix_sparsity:
        weighted_matrix_dense = prune_low_magnitude(weighted_matrix_dense, prune_weight_matrix_sparsity)

    fm = FieldWeightedFactorizationMachine(second_embeddings, field_embedding, weighted_matrix_dense)
    #deep
    if hidden_dims:
        if prune_deep_sparsity:
            deep = prune_low_magnitude(Deep_Prunable(hidden_dims, layer_prunables), prune_deep_sparsity)            
        else:
            deep = Deep_Prunable(hidden_dims, layer_prunables)
    #model flow
    fm_out, second_emb = fm(index_inp, coef_inp)
    
    if hidden_dims:  
        flatten_second_emb = keras.layers.Flatten()(second_emb) 
        deep_out = deep(flatten_second_emb)
        deep_out = tf.squeeze(deep_out)
        output = fm_out + deep_out
    else:
        output = fm_out
    output = keras.activations.sigmoid(output)
    model = keras.Model(inputs=[index_inp, coef_inp], outputs=output)
    return model
    

def create_deepfm_functional_model(n_categs, hidden_dims, layer_prunables, emb_dim,
                                   prune_emb_sparsity, prune_deep_sparsity, embedding_l2):
    '''create model by tensorflow function API
    
    Args:
    ----
    n_categs: List[int]. Number of distint categories in each field.
    
    '''
    #hidden_dims = [400,400,400]
    # layer_prunables = [False, True, True]
    
    #inputs
    index_inp = keras.layers.Input(shape=(None,), dtype=tf.int32, name='index')
    coef_inp =  keras.layers.Input(shape=(None,), dtype=tf.float32, name='coef')
    
    #embeddings:
    first_embeddings = []
    second_embeddings = []
    for n_categ in n_categs:
        first_emb = Embedding_Prunalbe(0, n_categ, 1,)
        sec_emb = Embedding_Prunalbe(0.1, n_categ, emb_dim, embedding_l2)
        if prune_emb_sparsity:
            first_emb = prune_low_magnitude(first_emb, prune_emb_sparsity)
            sec_emb = prune_low_magnitude(sec_emb, prune_emb_sparsity)
        first_embeddings.append(first_emb)
        second_embeddings.append(sec_emb) 

    #fm        
    fm = FactorizationMachines(first_embeddings, second_embeddings)

    #deep
    if prune_deep_sparsity:
        deep = prune_low_magnitude(Deep_Prunable(hidden_dims, layer_prunables), prune_deep_sparsity)
    else:
        deep = Deep_Prunable(hidden_dims, layer_prunables)
    
    # model 
    fm_out, second_emb = fm(index_inp, coef_inp)
    flatten_second_emb = keras.layers.Flatten()(second_emb)    
    deep_out = deep(flatten_second_emb)
    deep_out = tf.squeeze(deep_out)
    # output = keras.activations.sigmoid(fm_out + deep_out)
    output = keras.activations.sigmoid(fm_out + deep_out)
    model = keras.Model(inputs=[index_inp, coef_inp], outputs=output)
    return model


class DeepFM_Prunable(tf.keras.Model):
    def __init__(self, n_categs, emb_dim, hidden_dims, layer_prunables, 
                 prune_emb_sparsity, prune_deep_sparsity):
        super().__init__()
        self.first_embeddings = []
        self.second_embeddings = []
        self._create_embedding(n_categs, emb_dim)
        self.fm = FactorizationMachines(self.first_embeddings, self.second_embeddings)
        self.deep = prune_low_magnitude(
            Deep_Prunable(hidden_dims, layer_prunables), sparsity
        )
        self.flatten = keras.layers.Flatten()
        
    def call(self,index, coef):
        fm_out, second_emb = self.fm(index, coef)
        flatten_second_emb = self.flatten(second_emb)    
        deep_out = self.deep(flatten_second_emb)
        deep_out = tf.squeeze(deep_out)
        output = keras.activations.sigmoid(fm_out + deep_out)
        return output
        
        
    def _create_embedding(self, n_categs, emb_dim):
        for n_categ in n_categs:
            self.first_embeddings.append(Embedding_Prunalbe(0, n_categ, 1))
            self.second_embeddings.append(Embedding_Prunalbe(0.5, n_categ, emb_dim)) 
        
            