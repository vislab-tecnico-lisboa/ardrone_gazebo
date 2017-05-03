# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:41:08 2017

@author: Nino Cauli
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class AC_Network():
    def __init__(self,s_size,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None,s_size],dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs,shape=[-1, 9])
            print "after inputs"
#            self.imageIn = tf.reshape(self.inputs,shape=[-1, 84, 84, 3])
#            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
#                inputs=self.imageIn,num_outputs=16,
#                kernel_size=[8,8],stride=[4,4],padding='VALID')
#            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
#                inputs=self.conv1,num_outputs=32,
#                kernel_size=[4,4],stride=[2,2],padding='VALID')
#            hidden = slim.fully_connected(slim.flatten(self.conv2),256,activation_fn=tf.nn.elu)
            hidden = slim.fully_connected(slim.flatten(self.imageIn),256,activation_fn=tf.nn.elu)
            print "after hidden"
            
            #Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            #step_size = tf.shape(self.imageIn)[:1]
            step_size = tf.shape(self.inputs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])
            print "after recurrent"
            
            #Output layers for policy and value estimations
            self.policy_mean = slim.fully_connected(rnn_out,a_size,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.policy_std_dev = slim.fully_connected(rnn_out,a_size,
                activation_fn=tf.nn.softplus,
                weights_initializer=normalized_columns_initializer(0.01),
                biases_initializer=None)
            self.value = slim.fully_connected(rnn_out,1,
                activation_fn=None,
                weights_initializer=normalized_columns_initializer(1.0),
                biases_initializer=None)
                
            print "after out"
                
            self.dist = tf.contrib.distributions.Normal(mu=self.policy_mean, sigma=self.policy_std_dev)
            self.samp_action = self.dist.sample([1])
            
            print "after dist"
            
            
            #Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None,a_size],dtype=tf.float32)
                #self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None],dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None],dtype=tf.float32)
                
                print "after placeholders"
                
                self.pdfs = self.dist.pdf(self.actions)
                
                self.advantages_stacked = tf.reshape(tf.tile(self.advantages,tf.constant([a_size])), [tf.size(self.advantages),a_size])

                #self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                #Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value,[-1])))
                self.entropy = tf.reduce_sum(0.5 * (1.0 + tf.log(2.0 * tf.square(self.policy_std_dev) * np.pi)))
                #self.policy_loss = -tf.reduce_sum(tf.multiply(tf.log(self.pdfs),self.advantages_stacked))
                self.policy_loss = -tf.reduce_sum(tf.log(self.pdfs)*self.advantages_stacked)
                self.loss = 0.5 * self.value_loss + self.policy_loss #- self.entropy * 0.01
                
                print "after loss"

                #Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                print "after get collection"
                self.gradients = tf.gradients(self.loss,local_vars)
                print "after get grad"
                self.var_norms = tf.global_norm(local_vars)
                print "after var norm"
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
                
                print "after grad"
                
                #Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))
            
            print "after all"