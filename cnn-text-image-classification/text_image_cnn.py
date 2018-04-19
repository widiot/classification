import tensorflow as tf
import numpy as np


class TextImageCNN():
    def __init__(
            self,
            num_classes,
            sequence_length,
            vocab_size,
            embedding_size,
            filter_sizes,
            num_filters,
            l2_reg_lambda_end=0.0,
            image_size=28,
            num_channels=3,
            filter1_size=5,
            filter1_deep=32,
            filter2_size=5,
            filter2_deep=64,
            fc_size=512,
            l2_reg_lambda_im=0.0,
    ):
        # 输入层
        # ==================================================
        with tf.name_scope('input'):
            self.input_x_txt = tf.placeholder(
                tf.int32, [None, sequence_length], name='input_x_txt')
            self.input_x_im = tf.placeholder(
                tf.float32, [None, image_size, image_size, num_channels],
                name='input_x_im')
            self.input_y = tf.placeholder(
                tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')

        # 图像模型
        # ==================================================

        with tf.name_scope('conv1-im'):
            filter_shape = [
                filter1_size, filter1_size, num_channels, filter1_deep
            ]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[filter1_deep]), name='b')
            conv1 = tf.nn.conv2d(
                self.input_x_im,
                W,
                strides=[1, 1, 1, 1],
                padding='SAME',
                name='conv')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name='relu')

        with tf.name_scope('pool1-im'):
            pool1 = tf.nn.max_pool(
                relu1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool')

        with tf.name_scope('conv2-im'):
            filter_shape = [
                filter2_size, filter2_size, filter1_deep, filter2_deep
            ]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[filter2_deep]), name='b')
            conv2 = tf.nn.conv2d(
                pool1, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name='relu')

        with tf.name_scope('pool2-im'):
            pool2 = tf.nn.max_pool(
                relu2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool')

        with tf.name_scope('reshape-im'):
            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool2, [-1, nodes])

        with tf.name_scope('fc1-im'):
            W = tf.Variable(
                tf.truncated_normal([nodes, fc_size], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[fc_size]), name='b')
            if l2_reg_lambda_im:
                fc1_l2_loss = tf.contrib.layers.l2_regularizer(
                    l2_reg_lambda_im)(W)
                tf.add_to_collection('losses', fc1_l2_loss)
            fc1 = tf.nn.relu(tf.nn.xw_plus_b(reshaped, W, b), name='relu')

        with tf.name_scope('dropout-im'):
            self.fc1_drop = tf.nn.dropout(
                fc1, self.dropout_keep_prob, name='dropout')

        # 文本模型
        # ==================================================

        with tf.name_scope('embedding-txt'):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name='W')
            self.embedded_chars = tf.nn.embedding_lookup(
                self.W, self.input_x_txt)
            self.embedded_chars_expanded = tf.expand_dims(
                self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-txt-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(
                    tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(
                    tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        with tf.name_scope('concat-txt'):
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('dropout-txt'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                        self.dropout_keep_prob)

        # 组合层
        # ==================================================

        with tf.name_scope('concat'):
            self.feature_concat = tf.concat([self.h_drop, self.fc1_drop], 1)

        # 输出层
        # ==================================================

        with tf.name_scope('output'):
            W = tf.Variable(
                tf.truncated_normal(
                    [fc_size + num_filters_total, num_classes], stddev=0.1),
                name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            if l2_reg_lambda_end:
                W_l2_loss = tf.contrib.layers.l2_regularizer(
                    l2_reg_lambda_end)(W)
                tf.add_to_collection('losses', W_l2_loss)
            self.scores = tf.nn.xw_plus_b(
                self.feature_concat, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # 计算交叉损失熵
        with tf.name_scope('loss'):
            mse_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.input_y))
            tf.add_to_collection('losses', mse_loss)
            self.loss = tf.add_n(tf.get_collection('losses'))

        # 正确率
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, 'float'), name='accuracy')