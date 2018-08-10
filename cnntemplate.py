# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:11:56 2018

@author: menguan
"""

import tensorflow as tf
import numpy as np
import re
##tools function

## 权重变量
def weight_variable(shape, name = ''):
    initial = tf.truncated_normal(shape, stddev=0.01)
    if name != '':
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)
## 偏置 变量
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

#这种更好
def weight_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def bias_var(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

#清理文本
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

## 上采样 输入 input [batchsize,datasize,datzsize,features]
def upsample(input, size):
    return tf.image.resize_nearest_neighbor(input, size=(int(size), int(size)))

## 卷积层 改变厚度 输入 input [batchsize,datasize,datzsize,features]
def conv2d(input, in_features, out_features, kernel_size, st=1,with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.conv2d(input, W, [ 1, st, st, 1 ], padding='SAME')
    if with_bias:
        return tf.add(conv , bias_variable([ out_features ]))
    return conv

#平均池化层 
def avg_pool(input, s ,st=-1):
    if(st==-1):
        st=s;
    return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, st, st, 1 ], 'SAME')

#最大池化层 
def max_pool(input, s, st=-1):
    if(st==-1):
        st=s;
    return tf.nn.max_pool(input, [ 1, s, s, 1 ], [1, st, st, 1 ], 'SAME')



#全连接层
def fc(input, in_features, out_features,with_bias=True):
    input=tf.reshape(input, [ -1, in_features ])
    W = weight_variable([ in_features, out_features ])
    fcout=tf.matmul(input, W)
    if with_bias:
        return tf.add(fcout , bias_variable([ out_features ]))
    return fcout

#densenet block层 layers:block内层数 growth:每层增加厚度 根据需要进行调整
def block(input, layers, in_features, growth, is_training=False, keep_prob=1.0):
    current = input
    features = in_features
    for idx in range(layers):  
      
        #此处为单小层结构 自行修改
        
#        kernel_size=3
#        tmp = tf.contrib.layers.batch_norm(current, scale=True, is_training=is_training, updates_collections=None)
#        tmp = tf.nn.relu(tmp)
#        tmp = conv2d(tmp, features, growth, kernel_size)
#        tmp = tf.nn.dropout(tmp, keep_prob)
        
            #for densenet121
        tmp = conv2d(current, features, growth, 1, with_bias=True)
        tmp = conv2d(tmp, features, growth, 3 , with_bias=True)
            #
        
        ###
    
        current = tf.concat((current, tmp), axis=3)
        features += growth
    return current, features #返回block层结果 和 厚度





#basic structure




#例如 input = tf.placeholder(dtype=tf.float32,shape=[None, 224,224,3], name="input_images") now datasize=224
def vgg16(input, datasize):
    current=conv2d(input, 3, 64, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=conv2d(current, 64, 64, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=max_pool(current,2) #datasize/2
    current=conv2d(current, 64, 128, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=conv2d(current, 128, 128, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=max_pool(current,2) #datasize/4
    current=conv2d(current, 128, 256, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=conv2d(current, 256, 256, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=conv2d(current, 256, 256, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=max_pool(current,2) #datasize/8
    current=conv2d(current, 256, 512, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=conv2d(current, 512, 512, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=conv2d(current, 512, 512, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=max_pool(current,2) #datasize/16
    current=conv2d(current, 512, 512, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=conv2d(current, 512, 512, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=conv2d(current, 512, 512, 3,with_bias=True) ;current = tf.nn.relu(current)
    current=max_pool(current,2) #datasize/32
    shape = int(np.prod(current.get_shape()[1:]))
    current=fc(current,shape,4096,with_bias=True) ;current = tf.nn.relu(current)
    current=fc(current,4096,4096,with_bias=True) ;current = tf.nn.relu(current)
    current=fc(current,4096,1000,with_bias=True) 
    return current


def densenet121(input,datasize, is_training, keep_prob):
    #first change the block # haven't add enough activation function
    features=16
    current=conv2d(input, 3, features, 3, with_bias=True)
    current=conv2d(current, features, features, 7,st= 2 ,with_bias=True)
    current=max_pool(current,3,2)
    current, features = block(current, 6, features, 32, is_training, keep_prob)
    current= conv2d(current, features, features, 1 ,with_bias=True)
    current= avg_pool(current,2)
    current, features = block(current, 12, features, 32, is_training, keep_prob)
    current= conv2d(current, features, features, 1 ,with_bias=True)
    current= avg_pool(current,2)
    current, features = block(current, 24, features, 32, is_training, keep_prob)
    current= conv2d(current, features, features, 1 ,with_bias=True)
    current= avg_pool(current,2)
    current, features = block(current, 16, features, 32, is_training, keep_prob)
    current= avg_pool(current,7)
    shape = int(np.prod(current.get_shape()[1:]))
    current=fc(current,shape,1000,with_bias=True) ;current=tf.nn.softmax(current)
    return current













