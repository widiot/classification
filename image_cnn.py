import tensorflow as tf
import numpy as np


class ImageCNN():
    def __init__(self,
                 num_classes,
                 image_size=28,
                 num_channels=3,
                 filter1_size=5,
                 filter1_deep=32,
                 filter2_size=5,
                 filter2_deep=64,
                 fc_size=512,
                 l2_reg_lambda=0.0):
        # 输入层
        self.input_x = tf.placeholder(
            tf.float32, [None, image_size, image_size, num_channels],
            name='input_x')
        self.input_y = tf.placeholder(
            tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')

        # 第一层卷积层
        with tf.name_scope('conv1'):
            # filter_shape=[长，宽，输入深度，输出深度]
            filter_shape = [
                filter1_size, filter1_size, num_channels, filter1_deep
            ]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[filter1_deep]), name='b')
            conv1 = tf.nn.conv2d(
                self.input_x,
                W,
                strides=[1, 1, 1, 1],
                padding='SAME',
                name='conv')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name='relu')

        # 第二层池化层
        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(
                relu1,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool')

        # 第三层卷积层
        with tf.name_scope('conv2'):
            filter_shape = [
                filter2_size, filter2_size, filter1_deep, filter2_deep
            ]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[filter2_deep]), name='b')
            conv2 = tf.nn.conv2d(
                pool1, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name='relu')

        # 第四层池化层
        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(
                relu2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool')

        # 将输出转为全连接层输入
        with tf.name_scope('reshape'):
            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool2, [-1, nodes])  # -1能自动推导出正确维数

        # 第五层全连接层
        with tf.name_scope('fc1'):
            W = tf.Variable(
                tf.truncated_normal([nodes, fc_size], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[fc_size]), name='b')
            # 正则化
            if l2_reg_lambda:
                fc1_l2_loss = tf.contrib.layers.l2_regularizer(l2_reg_lambda)(
                    W)
                tf.add_to_collection('losses', fc1_l2_loss)
            fc1 = tf.nn.relu(tf.nn.xw_plus_b(reshaped, W, b), name='relu')

        # dropout
        with tf.name_scope('dropout'):
            fc1_drop = tf.nn.dropout(
                fc1, self.dropout_keep_prob, name='dropout')

        # 第六层全连接层
        with tf.name_scope('output'):
            W = tf.Variable(
                tf.truncated_normal([fc_size, num_classes], stddev=0.2),
                name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            if l2_reg_lambda:
                fc2_l2_loss = tf.contrib.layers.l2_regularizer(l2_reg_lambda)(
                    W)
                tf.add_to_collection('losses', fc2_l2_loss)
            self.scores = tf.nn.xw_plus_b(fc1_drop, W, b, name='scores')
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
            correct_prediction = tf.equal(self.predictions,
                                          tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, 'float'), name='accuracy')
