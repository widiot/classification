import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# 参数
# ==================================================

flags = tf.flags

# 数据加载参数
flags.DEFINE_float('dev_sample_percentage', 0.1,
                   'Percentage of the training data to use for validation')
flags.DEFINE_string(
    'data_files',
    './data/maildata/spam_5000.utf8,./data/maildata/ham_5000.utf8',
    'Comma-separated data source files')

# 模型超参数
flags.DEFINE_integer('embedding_dim', 128,
                     'Dimensionality of character embedding (default: 128)')
flags.DEFINE_string('filter_sizes', '3,4,5',
                    'Comma-separated filter sizes (default: "3,4,5")')
flags.DEFINE_integer('num_filters', 128,
                     'Number of filters per filter size (default: 128)')
flags.DEFINE_float('dropout_keep_prob', 0.5,
                   'Dropout keep probability (default: 0.5)')
flags.DEFINE_float('l2_reg_lambda', 0.0,
                   'L2 regularization lambda (default: 0.0)')

# 训练参数
flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
flags.DEFINE_integer('num_epochs', 10,
                     'Number of training epochs (default: 10)')
flags.DEFINE_integer(
    'evaluate_every', 100,
    'Evaluate model on dev set after this many steps (default: 100)')
flags.DEFINE_integer('checkpoint_every', 100,
                     'Save model after this many steps (default: 100)')
flags.DEFINE_integer('num_checkpoints', 5,
                     'Number of checkpoints to store (default: 5)')

# 其他参数
flags.DEFINE_boolean('allow_soft_placement', True,
                     'Allow device soft device placement')
flags.DEFINE_boolean('log_device_placement', False,
                     'Log placement of ops on devices')

FLAGS = flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('')

# 数据准备
# ==================================================

# 加载数据
print('Loading data...')
x_text, y = data_helpers.load_data_and_labels(FLAGS.data_files)

# 建立词汇表
max_document_length = max([len(x.split(' ')) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# 随机混淆数据
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# 划分train/test数据集
# TODO: 这种做法比较暴力，应该用交叉验证
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled

print('Vocabulary Size: {:d}'.format(len(vocab_processor.vocabulary_)))
print('Train/Dev split: {:d}/{:d}'.format(len(y_train), len(y_dev)))
print('')

# 训练
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # 定义训练相关操作
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # 跟踪梯度值和稀疏性（可选）
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    '{}/grad/hist'.format(v.name), g)
                sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(
                    v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 模型和摘要的保存目录
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, 'runs', timestamp))
        print('\nWriting to {}\n'.format(out_dir))

        # 损失值和正确率的摘要
        loss_summary = tf.summary.scalar('loss', cnn.loss)
        acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

        # 训练摘要
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                     sess.graph)

        # 开发摘要
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # 检查点目录，默认存在
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # 写入词汇表文件
        vocab_processor.save(os.path.join(out_dir, 'vocab'))

        # 初始化变量
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            一个训练步骤
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run([
                train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy
            ], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(
                time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            在开发集上验证模型
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(
                time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

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
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print('')
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(
                    sess, checkpoint_prefix, global_step=current_step)
                print('Saved model checkpoint to {}\n'.format(path))