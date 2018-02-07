import tensorflow as tf
import numpy as np


class ImageCNN():
    # 图像参数
    IMAGE_SIZE = 100
    NUM_CHANNELS = 3

    # 第一层卷积参数
    FILTER1_SIZE = 5
    FILTER1_DEEP = 32

    # 第二层卷积参数
    FILTER2_SIZE = 5
    FILTER2_DEEP = 64

    # 全连接层结点数
    FC_SIZE = 512

    def __init__(self, num_classes, l2_reg_lambda=0.0):
        # 输入层
        self.input_x = tf.placeholder(
            tf.int32,
            [None, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS],
            name='input_x')
        self.input_y = tf.placeholder(
            tf.int32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name='dropout_keep_prob')

        # 第一层卷积层
        with tf.name_scope('conv1'):
            # filter_shape=[长，宽，输入深度，输出深度]
            filter_shape = [
                self.FILTER1_SIZE, self.FILTER1_SIZE, self.NUM_CHANNELS,
                self.FILTER1_DEEP
            ]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(
                tf.constant(0.1, shape=[self.FILTER1_DEEP]), name='b')
            conv1 = tf.nn.con2d(
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
                name='pool1')

        # 第三层卷积层
        with tf.name_scope('conv2'):
            filter_shape = [
                self.FILTER2_SIZE, self.FILTER2_SIZE, self.FILTER1_DEEP,
                self.FILTER2_DEEP
            ]
            W = tf.Variable(
                tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, [self.FILTER1_DEEP]), name='b')
            conv2 = tf.nn.con2d(
                pool1, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name='relu')

        # 第四层池化层
        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(
                relu2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool2')

        # 将输出转为全连接层输入
        with tf.name_scope('reshape'):
            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

        # 第五层全连接层
        with tf.name_scope('fc1'):
            W = tf.Variable(
                tf.truncated_normal([nodes, self.FC_SIZE], stddev=0.1),
                name='W')
            b = tf.Variable(tf.constant(0.1, [self.FC_SIZE]))
            # 正则化
            if l2_reg_lambda:
                fc1_l2_loss = tf.contrib.layers.l2_regularizer(l2_reg_lambda)(
                    W)
                tf.add_to_collection('losses', fc1_l2_loss)
            fc1 = tf.nn.relu(tf.matmul(reshaped, W) + b)
            # dropout
            fc1 = tf.nn.dropout(fc1, self.dropout_keep_prob)

        # 第六层全连接层
        with tf.name_scope('output'):
            W = tf.Variable(
                tf.truncated_normal([self.FC_SIZE, num_classes], stddev=0.2),
                name='W')
            b = tf.Variable(tf.constant(0.1, [num_classes]), name='b')
            if l2_reg_lambda:
                fc2_l2_loss = tf.contrib.layers.l2_regularizer(l2_reg_lambda)(
                    W)
                tf.add_to_collection('losses', fc2_l2_loss)
            self.scores = tf.nn.xw_plus_b(fc1, W, b, name='scores')
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
