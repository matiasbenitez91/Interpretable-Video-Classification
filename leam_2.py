# -*- coding: utf-8 -*-
"""
Guoyin Wang

LEAM
"""

import os, sys, pickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from math import floor

from model import *
from utils import get_minibatches_idx, restore_from_save, tensors_key_in_file, prepare_data_for_emb, load_class_embedding

class Options(object):
    def __init__(self):
        self.GPUID = 0
        self.dataset = 'yelp_full'
        self.fix_emb = True
        self.restore = False
        self.W_emb = None
        self.W_class_emb = None
        self.maxlen = 10
        self.n_words = None
        self.embed_size = 2048
        self.lr = 1e-3
        self.batch_size = 100
        self.max_epochs = 150
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0 
        self.save_path = "./save_model_11_2/"
        self.log_path = "./log/"
        self.print_freq = 100
        self.valid_freq = 5

        self.optimizer = 'Adam'
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 55
        self.H_dis = 200


    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

def emb_classifier(x, x_mask, y, dropout, opt, class_penalty):
    # comment notation
    #  b: batch size, s: sequence length, e: embedding dim, c : num of class
    #x_emb, W_norm = embedding(x, opt)  #  b * s * e
    x_emb=x
    x_emb=tf.cast(x_emb,tf.float32)
    #W_norm=tf.cast(W_norm,tf.float32)
    y_pos = tf.argmax(y, -1)
    #y_emb, W_class = embedding_class(y_pos, opt, 'class_emb') # b * e, c * e
    #y_emb=tf.cast(y_emb,tf.float32)
    W_class=opt.W_class_emb
    W_class=tf.cast(W_class,tf.float32)
    W_class_tran = tf.transpose(W_class, [1,0]) # e * c
    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
    H_enc , beta= att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt)
    #H_enc = tf.squeeze(H_enc)
    # H_enc=tf.cast(H_enc,tf.float32)
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=False)  # b * c
    logits_class = discriminator_2layer(W_class, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
    prob = tf.nn.softmax(logits)
    class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32, value=np.identity(opt.num_class),)
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) + class_penalty * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=class_y, logits=logits_class))

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opt.optimizer,
        learning_rate=opt.lr)

    return accuracy, loss, train_op, global_step, H_enc, beta, prob


def main():
    # Prepare training and testing data
    with open('data_new/train_11.p', 'rb') as f:
        train=pickle.load(f)
    with open('data_new/test_11.p', 'rb') as f:
        val=pickle.load(f)
        
    train_lab=train[1]
    train=train[0]


    val_lab=val[1]
    val=val[0]

    with open('proto_ucf11.p', 'rb') as f:
        class_=pickle.load(f)


    # In[3]:

    list_train_acc=[]
    list_val_acc=[]
    class_=class_[0]
    opt = Options()
    opt.num_class=11
    # load data
    try:
        opt.W_emb = None
        opt.W_class_emb =  class_#load_class_embedding( wordtoix, opt)
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[None, opt.maxlen, opt.embed_size],name='x_')
        x_mask_ = tf.placeholder(tf.float32, shape=[None, opt.maxlen],name='x_mask_')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        y_ = tf.placeholder(tf.float32, shape=[None, opt.num_class],name='y_')
        class_penalty_ = tf.placeholder(tf.float32, shape=())
        accuracy_, loss_, train_op, global_step, H_enc_, beta_, prob_= emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
    uidx = 0
    max_val_accuracy = 0.
    max_test_accuracy = 0.

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                t_vars = tf.trainable_variables()
                save_keys = tensors_key_in_file(opt.save_path)
                ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
                cc = {var.name: var for var in t_vars}
                # only restore variables with correct shape
                ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

                loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
                loader.restore(sess, opt.save_path)

                print("Loading variables from '%s'." % opt.save_path)
                print("Loaded variables:" + str(ss))

            except:
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(opt.max_epochs):
                print("Starting epoch %d" % epoch)
                kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
                for _, train_index in kf:
                    uidx += 1
                    sents = [train[t] for t in train_index]
                    sents=np.array(sents)
                    x_labels = [train_lab[t] for t in train_index]
                    x_labels = np.array(x_labels)
                    x_labels = x_labels.reshape((len(x_labels), opt.num_class))

                    x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)
                    _, loss, step,  = sess.run([train_op, loss_, global_step], feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels, keep_prob: opt.dropout, class_penalty_:opt.class_penalty})

                if epoch % opt.valid_freq == 0:
                    train_correct = 0.0
                    # sample evaluate accuaccy on 500 sample data
                    kf_train = get_minibatches_idx(500, opt.batch_size, shuffle=True)
                    for _, train_index in kf_train:
                        train_sents = [train[t] for t in train_index]
                        train_labels = [train_lab[t] for t in train_index]
                        train_labels = np.array(train_labels)
                        train_sents=np.array(train_sents)
                        train_labels = train_labels.reshape((len(train_labels), opt.num_class))
                        x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, opt)  
                        train_accuracy = sess.run(accuracy_, feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask, y_: train_labels, keep_prob: 1.0, class_penalty_:0.0})

                        train_correct += train_accuracy * len(train_index)

                    train_accuracy = train_correct / 500
                    list_train_acc.append(train_accuracy)
                    print("Iteration %d: Training loss %f " % (uidx, loss))
                    print("Train accuracy %f " % train_accuracy)

                    val_correct = 0.0
                    kf_val = get_minibatches_idx(len(val), len(val), shuffle=True)
                    for _, val_index in kf_val:
                        val_sents = [val[t] for t in val_index]
                        val_labels = [val_lab[t] for t in val_index]
                        val_labels = np.array(val_labels)
                        val_labels = val_labels.reshape((len(val_labels), opt.num_class))
                        val_sents=np.array(val_sents)
                        x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, opt)
                        val_accuracy = sess.run(accuracy_, feed_dict={x_: x_val_batch, x_mask_: x_val_batch_mask,
                            y_: val_labels, keep_prob: 1.0,
                            class_penalty_:0.0                                         })

                        val_correct += val_accuracy * len(val_index)

                    val_accuracy = val_correct / len(val)
                    list_val_acc.append(val_accuracy)
                    print("Validation accuracy %f " % val_accuracy)

                    if val_accuracy > max_val_accuracy:
                        max_val_accuracy = val_accuracy

                        test_correct = 0.0
                        saver.save(sess, opt.save_path+'model'+str(opt.num_class)+'.ckpt')
                
            print("Max val accuracy %f " % max_val_accuracy)
            with open('example11.p', 'rb') as f:
                example=pickle.load(f)
            x_batch, x_batch_mask = prepare_data_for_emb(example, opt)
            h_enc=sess.run([beta_, prob_], feed_dict={x_:x_batch, x_mask_:x_batch_mask, keep_prob: 1.0, class_penalty_:0.0})
            #print('H_ENC_', h_enc.shape)
            pickle.dump(h_enc, open('beta_example11_.p','wb'))
            pickle.dump([list_train_acc, list_val_acc], open('list_acc_11_.p','wb'))
        except KeyboardInterrupt:
            print('Training interupted')
            print("Max val accuracy %f " % max_val_accuracy)

            
main()