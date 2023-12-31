import math as m
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

# Row wise RMSE
def rwrmse(y_true, y_pred):
	return tf.reduce_mean(K.pow(tf.reduce_mean(K.pow(y_true - y_pred, 2), axis=1), 0.5), axis=-1)

# Custom logistic loss
def logistic_loss(y_true, y_pred):
	return tf.reduce_mean(K.log(1. + K.exp(-(y_true * y_pred))))

# Exponential sign loss 
def exp_sign(y_true, y_pred):
	return tf.reduce_mean(K.exp(-(y_true * y_pred)))

# Cosine dis-similarity
def rw_cosine_dissimilarity(y_true, y_pred):
	inner_prod = K.sum(tf.math.multiply(y_true, y_pred), axis=1)
	norm_prod = tf.multiply(tf.norm(y_true, ord=2, axis=1), tf.norm(y_pred, ord=2, axis=1))
	return tf.reduce_mean(inner_prod/norm_prod) + 1. # Add 1 to make loss range bounded between [0, 2]

# Multi-head rmrmse
def multi_rwrmse(y_true, y_pred):
	# Chunk split
	n = 3
	# Split y_pred into chunks
	chunks = tf.split(y_pred, num_or_size_splits=n, axis=1)
	# Loss accumulator
	loss = 0.
	for i in range(len(chunks)):
		loss += rwrmse(y_true, chunks[i])
	return loss / n

# Pseudo-huber loss (continuous huber) alpha = 1.0
def pseudo_huber(y_true, y_pred):
	diff = y_true - y_pred
	loss = K.pow(K.pow(diff, 2) + 1., 0.5) - 1.
	return tf.reduce_mean(loss)

# Cauchy loss, alpha = 0.
def cauchy(y_true, y_pred):
	c = 1.0
	diff = y_true - y_pred
	loss = K.log(0.5 * K.pow(diff / c, 2) + 1.)
	return tf.reduce_mean(loss)

# Geman-Mcclure loss, alpha = -2.
def geman_mcclure(y_true, y_pred):
	diff = y_true - y_pred
	loss = 2 * K.pow(diff, 2) * K.pow(K.pow(diff, 2) + 4., -1)
	return tf.reduce_mean(loss)
