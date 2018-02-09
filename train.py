import tensorflow as tf
import numpy as np

# 参数
# ==================================================

flags = tf.flags

# 数据加载参数
flags.DEFINE_string('data_directory', './data/flower_photos/',
                    'Directory path of the image data')
flags.DEFINE_float('validation_percentage', 0.1,
                   'Percentage of the image data to use for validation')
flags.DEFINE_float('test_percentage', 0.1,
                   'Percentage of the image data to use for test')

# 模型超参数
flags.DEFINE_float('learning_rate_base', 0.01,
                   'Base leaning rate for exponential decay')
flags.DEFINE_float('learning_rate_decay', 0.99,
                   'Attenuation rate of learning rate')
flags.DEFINE_float('l2_reg_lambda', 0.0001, 'Lambda for L2 regularizer')
flags.DEFINE_float('moving_average_decay', 0.99,
                   'Attenuation rate of moving average')

# 训练参数
flags.DEFINE_integer('batch_size', 100, 'Number of image in a batch')
flags.DEFINE_integer('num_epochs', 1, 'Number of training epoch')
