# Recommendation and Pruning (under development)

Implement some Recommendation model (deepFM, deepFwFm) and try model pruning using TFMOT(tensorflow_model_optimization).

## Current Issues:
1. Custom layer failed when save model.  
`tf.function` signature is necessary when save model, while add `tf.function` will raise error when training with `pruning_wrapper.PruneLowMagnitude`. This error is raised from `inspect.getfullargspec(layer.call)`, might be fixed with few code change.

2. ML1M is too easy to train (over parameterized?), cause metrics barely drop while pruning. Switch to criteo...

## Model Reference:
1. DeepLight: Deep Lightweight Feature Interactions for Accelerating CTR Predictions in Ad Serving. (WSDM'21)
2. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction. (IJCAI 2017)

## Environment:
* python 3.6.10
* tensorflow==2.3.0
* tensorflow-model-optimization==0.5.0

