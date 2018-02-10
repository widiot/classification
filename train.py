import tensorflow as tf
import numpy as np
import data_helper
from image_cnn import ImageCNN

# 参数
# ==================================================

flags = tf.flags

# 数据参数
flags.DEFINE_string('data_directory', './data/flower_photos/',
                    'Directory path of the image data')
flags.DEFINE_float('validation_percentage', 0.1,
                   'Percentage of the training data to use for validation')
flags.DEFINE_integer('image_size', 100, 'Image size for resizing')
flags.DEFINE_integer('num_channels', 3, 'Number of image channels')

# 模型超参数
flags.DEFINE_float('learning_rate_base', 0.01,
                   'Base leaning rate for exponential decay')
flags.DEFINE_float('learning_rate_decay', 0.99,
                   'Attenuation rate of learning rate')
flags.DEFINE_float('l2_reg_lambda', 0.0001, 'Lambda for L2 regularizer')
flags.DEFINE_float('moving_average_decay', 0.99,
                   'Attenuation rate of moving average')
flags.DEFINE_integer('filter1_size', 5, 'Size of the first filter')
flags.DEFINE_integer('filter1_deep', 32, 'Deep of the first filter')
flags.DEFINE_integer('filter2_size', 5, 'Size of the second filter')
flags.DEFINE_integer('filter2_deep', 64, 'Deep of the second filter')
flags.DEFINE_integer('fc_size', 512, 'Size of the full conllection layer')

# 训练参数
flags.DEFINE_integer('batch_size', 64, 'Number of image in a batch')
flags.DEFINE_integer('num_epochs', 1, 'Number of training epoch')

# 打印参数
FLAGS = flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{} = {}'.format(attr.upper(), value))
print('')

# 训练
# ==================================================

# 加载数据
x_train, x_valid, y_train, y_valid = data_helper.load_train_valid_data(
    FLAGS.data_directory, FLAGS.validation_percentage)

# 创建模型对象
cnn = ImageCNN(
    image_size=FLAGS.image_size,
    num_channels=FLAGS.num_channels,
    filter1_size=FLAGS.filter1_size,
    filter1_deep=FLAGS.filter1_deep,
    filter2_size=FLAGS.filter2_size,
    filter2_deep=FLAGS.filter2_deep,
    fc_size=FLAGS.fc_size,
    num_classes=y_train.shape[1],
    l2_reg_lambda=FLAGS.l2_reg_lambda)
