import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import l2_regularizer, xavier_initializer, fully_connected, flatten

#various implementations of this exist, particularly in how the negatives are calculated
#for more, read https://arxiv.org/abs/1502.01852
def parametric_relu(X, regularizer=None, name="parametric_relu"):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alphas',
                                 regularizer=regularizer,
                                 dtype=X.dtype,
                                 shape=X.get_shape().as_list()[-1], 
                                 initializer=tf.constant_initializer(0.01))
        positives = tf.nn.relu(X)
        negatives = alphas*(tf.subtract(X, tf.abs(X))) * 0.5
        return positives + negatives

#this implementation assumes alpha will be less than 1. It would be stupid if it weren't.
def leaky_relu(X, alpha=0.2):
    return tf.maximum(X, alpha*X)

#https://arxiv.org/abs/1302.4389
#This doesn't need a scope. There aren't trainable parameters here. It's just a pool
def maxout(X, num_maxout_units, axis=None):
    input_shape = X.get_shape().as_list()
    
    axis = -1 if axis is None else axis
    
    num_filters = input_shape[axis]
    
    if num_filters % num_maxout_units != 0:
        raise ValueError, "num filters (%d) must be divisible by num maxout units (%d)" % (num_filters, num_maxout_units)
    
    output_shape = input_shape.copy()
    output_shape[axis] = num_maxout_units
    output_shape += [num_filters // num_maxout_units]
    return tf.reduce_max(tf.reshape(X, output_shape), -1, keep_dims=False)

def conv(X,
         output_filter_size,
         kernel=[5,5],
         strides=[2,2],
         w_initializer=xavier_initializer(),
         regularizer=None,
         name="conv"):
    
    with tf.variable_scope(name):
        W = tf.get_variable('W_conv',
                            regularizer=regularizer,
                            dtype=X.dtype,
                            shape=[kernel[0], kernel[1], X.get_shape().as_list()[-1], output_filter_size],
                            initializer=w_initializer)
        b = tf.get_variable('b_conv',
                            regularizer=regularizer,
                            dtype=X.dtype,
                            shape=[output_filter_size],
                            initializer=tf.zeros_initializer(dtype=X.dtype))
        
        return tf.nn.bias_add( tf.nn.conv2d(X,
                                            W,
                                            strides=[1,strides[0],strides[1],1],
                                            padding='SAME',
                                            name="conv2d"), b)

#get the convolutional output size given same padding and equal strides
def conv_out_size(sz, stride):
    return int(np.ceil(float(sz) / float(stride)))
