import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from itertools import chain
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_wrapper import PruneLowMagnitude


class FieldWeightedFactorizationMachine(keras.layers.Layer):
    def __init__(self, feature_embeddings, field_embedding, weighted_matrix_dense):
        '''
        arguments:
            feature_embeddings: List[Embedding]. the embedding for each field. 
                                len(feature_embeddings) should equal to number_of_field.
                                ith embedding have shape (number_of_category[i], embedding_dimension).
            field_embedding: Embedding with shape (number_of_field, embedding_dimension)
            weighted_matrix_dense: Dense. A dense layer with units equals to number_of_field.
                                   Note that you could set use_bias=False.
        '''
        super().__init__()
        # assert field_embedding.input_dim == len(feature_embeddings) == weighted_matrix_dense.units, \
        #        'input_dim of field_embedding and units of weighted_matrix_dense and number of feature_embeddings should be equal'
        self.w0 = tf.Variable(tf.initializers.GlorotNormal()((1,)))
        self._weighted_matrix_dense = weighted_matrix_dense
        self._num_field = len(feature_embeddings)
        self.weighted_matrix = None # num_field x num_field
        self.field_embedding = field_embedding # num_field x emb_dim
        self.feature_embeddings = feature_embeddings # List[embedding]

    def build(self, input_shape):
        self._weighted_matrix_dense.build((self._num_field,))
        if isinstance(self._weighted_matrix_dense, PruneLowMagnitude):
            self.weighted_matrix = self._weighted_matrix_dense.layer.kernel
        else:
            self.weighted_matrix = self._weighted_matrix_dense.kernel
        
        for emb in self.feature_embeddings:        
            if not emb.weights:
                emb.build(input_shape=tuple())
        if not self.field_embedding.weights:
            self.field_embedding.build(tuple())
        if not self._weighted_matrix_dense.weights:
            self._weighted_matrix_dense.build((self._weighted_matrix_dense.units,))

    def get_config(self,):
        return {
            'num_field': self._num_field,
            'embedding_dim': self.field_embedding.output_dim
        }

    @tf.function(input_signature=[
        tf.TensorSpec([None, None], tf.int32), 
        tf.TensorSpec([None, None], tf.float32),
    ]) 
    def call(self, index, coef):
        #coef: batch x num_field
        batch_w0 = tf.tile(self.w0, tf.shape(index)[0:1])
        feature_embs = tf.stack(
            [emb(index[:,no]) for no, emb in enumerate(self.feature_embeddings)], 
            axis=1            
        ) # batch x n_field x emb_dim
        feature_field_interaction = (
            feature_embs * self.field_embedding.weights[0] * tf.expand_dims(coef, axis=2)
        )
        feature_field_interaction = tf.reduce_sum(feature_field_interaction, [1,2]) # batch
        
        coef_feature_embs = feature_embs * tf.expand_dims(coef, 2) # batch x n_field x emb_dim
        outer = tf.einsum('ijk, ilk ->ijlk', coef_feature_embs, coef_feature_embs) # batch x n_field x n_field x emb_dim
        # batch x n_field x n_field
        all_interaction = tf.reduce_sum(outer, -1) 
        symmetric_weight_matrix = (self.weighted_matrix + tf.transpose(self.weighted_matrix))/2
        weighted_all_interaction = all_interaction * symmetric_weight_matrix
        # batch
        weighted_interaction = tf.reduce_sum(weighted_all_interaction, (1,2)) - tf.reduce_sum(tf.linalg.diag_part(weighted_all_interaction),-1)
        weighted_interaction = weighted_interaction / 2 
        output = batch_w0 + feature_field_interaction + weighted_interaction
        return output, coef_feature_embs


class FactorizationMachines(keras.layers.Layer):
    'factorization machines part for DeepFM. Note that deep and fm share same embedding layer'
    def __init__(self, first_embs, second_embs):
        '''
        Arguments:
        ----
        first_embs : NFieldEmbedding
        second_embs : NFieldEmbedding
        '''
        super().__init__()
        self.w0 = tf.Variable(tf.initializers.GlorotNormal()((1,)))
        self.first_embs =first_embs
        self.second_embs = second_embs
        self.fm_layer = FM_Order2()        

    def build(self, input_shape):
        super().build(input_shape)
        for emb in self.first_embs:
            if not emb.weights:
                emb.build(input_shape=tuple())
        for emb in self.second_embs:
            if not emb.weights:
                emb.build(input_shape=tuple())

    
    @tf.function(input_signature=[
        tf.TensorSpec([None, None], tf.int32), 
        tf.TensorSpec([None, None], tf.float32),
    ]) 
    def call(self, index, coef):
        #indexs shape : batch x num_field 
        #coef shape: batch x num_field 
        #ref : Factorization Machines, equation 1,3   
        batch_w0 = tf.tile(self.w0, tf.shape(index)[0:1])
        first = tf.concat(
            [emb(index[:,no]) for no, emb in enumerate(self.first_embs)], 
            axis=1
        ) # batch x n_field
        second_emb = tf.stack(
            [emb(index[:,no]) for no, emb in enumerate(self.second_embs)], 
            axis=1            
        ) # batch x n_field x emb_dim
        first = first * coef
        first = tf.reduce_sum(first, (1,)) # batch
        second = self.fm_layer(second_emb, coef) # batch
        score = batch_w0 + first + second 
        return score, second_emb    
    
    
class FM_Order2(keras.layers.Layer):
    'conduct order-2 factorization machines calcuation'

    def __init__(self, **kwargs):
        super(FM_Order2, self).__init__(**kwargs)

    def build(self, input_shape):
        super(FM_Order2, self).build(input_shape)
    
    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None], tf.float32, name='second_embeddings'), 
        tf.TensorSpec([None, None], tf.float32, name='coef'),
    ])    
    def call(self, second_emb, coef): 
        #second_emb : b x num_field x emb_dim    
        #coef : batch x num_feature
        #reference : Factorization Machines, lemma 3.1
        if isinstance(coef, tf.Tensor):
            coef_expand = tf.expand_dims(coef, 2) # b x num_field x 1
            second_1 = tf.reduce_sum(
                tf.reduce_sum(second_emb * coef_expand,(1,)) ** 2, 1
            ) # b 
            second_2 = tf.matmul(
                tf.expand_dims(coef, 1) ** 2, second_emb ** 2
            ) # b x 1 x emb_dim
            second_2 = tf.reduce_sum(second_2,(1, 2)) # b
        else:
            second_1 = tf.reduce_sum(tf.reduce_sum(second_emb, 1)**2, 1) # b 
            second_2 = tf.reduce_sum(second_emb**2, (1, 2))

        return 0.5*(second_1-second_2) # b x num_features x emb_features