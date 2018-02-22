import tensorflow as tf
import numpy as np
import time, os, datetime
import data_helpers
from image_cnn import ImageCNN

# 参数
# ==================================================

flags = tf.flags

# 数据参数
flags.DEFINE_string('data_directory', './data/flower_photos/',
                    'Directory path of the image data')
flags.DEFINE_float('validation_percentage', 0.1,
                   'Percentage of the training data to use for validation')
flags.DEFINE_integer('image_size', 28, 'Image size for resizing')
flags.DEFINE_integer('num_channels', 3, 'Number of image channels')

# 模型超参数
flags.DEFINE_float('learning_rate_base', 0.01,
                   'Base leaning rate for exponential decay')
flags.DEFINE_float('learning_rate_decay', 0.99,
                   'Attenuation rate of learning rate')
flags.DEFINE_float('l2_reg_lambda', 0.0001, 'Lambda for L2 regularizer')
flags.DEFINE_float('moving_average_decay', 0.99,
                   'Attenuation rate of moving average')
flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability')
flags.DEFINE_integer('filter1_size', 5, 'Size of the first filter')
flags.DEFINE_integer('filter1_deep', 32, 'Deep of the first filter')
flags.DEFINE_integer('filter2_size', 5, 'Size of the second filter')
flags.DEFINE_integer('filter2_deep', 64, 'Deep of the second filter')
flags.DEFINE_integer('fc_size', 512, 'Size of the full conllection layer')
flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store')
flags.DEFINE_integer('evaluate_every', 100,
                     'Evaluate model on dev set after this many steps')
flags.DEFINE_integer('checkpoint_every', 100,
                     'Save model after this many steps')

# 训练参数
flags.DEFINE_integer('batch_size', 100, 'Number of image in a batch')
flags.DEFINE_integer('num_epochs', 1000, 'Number of training epoch')

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
x_train, x_valid, y_train, y_valid = data_helpers.load_train_valid_data(
    FLAGS.data_directory, FLAGS.validation_percentage, FLAGS.image_size)

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

# 全局步数
global_step = tf.Variable(0, name='global_step', trainable=False)

# 滑动平均
with tf.name_scope('moving-average'):
    variable_average = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

# 指数衰减
with tf.name_scope('exponential-decay'):
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate_base, global_step,
        int(len(y_train) / FLAGS.batch_size), FLAGS.learning_rate_decay)

# 同时反向传播和滑动平均
optimizer_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    cnn.loss, global_step=global_step)  # 最小化损失函数
with tf.control_dependencies([optimizer_op, variable_average_op]):
    train_op = tf.no_op(name='train')

with tf.Session() as sess:

    # 模型和摘要的保存目录
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
    print('\nWriting to {}\n'.format(out_dir))

    # 损失值和正确率的摘要
    loss_summary = tf.summary.scalar('loss', cnn.loss)
    acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

    # 训练和验证摘要
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, 'summary', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    valid_summary_op = tf.summary.merge([loss_summary, acc_summary])
    valid_summary_dir = os.path.join(out_dir, 'summary', 'valid')
    valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

    # 检查点
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch):
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
            feed_dict)
        train_summary_writer.add_summary(summaries, step)
        time_str = datetime.datetime.now().isoformat()
        print('{}: step {}, loss {:g}, acc {:g}'.format(
            time_str, step, loss, accuracy))

    def valid_step(x_batch, y_batch, writer=None):
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, valid_summary_op, cnn.loss, cnn.accuracy], feed_dict)
        if writer:
            writer.add_summary(summaries, step)
        time_str = datetime.datetime.now().isoformat()
        print('{}: step {}, loss {:g}, acc {:g}'.format(
            time_str, step, loss, accuracy))

    # 生成batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    # 迭代训练每个batch
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print('\nEvaluation:')
            valid_step(x_valid, y_valid, writer=valid_summary_writer)
            print('')
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(
                sess, checkpoint_prefix, global_step=current_step)
            print('Saved checkpoint to {}\n'.format(path))
