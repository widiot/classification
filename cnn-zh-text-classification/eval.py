import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import csv
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# 参数
# ==================================================

flags = tf.flags

# 数据参数
flags.DEFINE_string(
    'data_files',
    './data/maildata/spam_5000.utf8,./data/maildata/ham_5000.utf8',
    'Comma-separated data source files')

# 评估参数
flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
flags.DEFINE_string('checkpoint_dir', './runs/1517572900/checkpoints',
                    'Checkpoint directory from training run')
flags.DEFINE_boolean('eval_train', False, 'Evaluate on all training data')

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

# 加载训练数据或者修改测试句子
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.data_files)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = [
        '亲爱的CFer，您获得了英雄级道具。还有全新英雄级道具在等你来拿，立即登录游戏领取吧！',
        '第一个build错误的解决方法能再说一下吗，我还是不懂怎么解决', '请联系张经理获取最新资讯'
    ]
    y_test = [0, 1, 0]

# 对自己的数据的处理
x_raw_cleaned = [
    data_helpers.clean_str(data_helpers.seperate_line(line)) for line in x_raw
]
print(x_raw_cleaned)

# 将数据转为词汇表的索引
vocab_path = os.path.join(FLAGS.checkpoint_dir, '..', 'vocab')
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw_cleaned)))

print('\nEvaluating...\n')

# 评估
# ==================================================

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 加载保存的元图和变量
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 通过名字从图中获取占位符
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        # input_y = graph.get_operation_by_name('input_y').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name(
            'dropout_keep_prob').outputs[0]

        # 我们想要评估的tensors
        predictions = graph.get_operation_by_name(
            'output/predictions').outputs[0]

        # 生成每个轮次的batches
        batches = data_helpers.batch_iter(
            list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # 收集预测值
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {
                input_x: x_test_batch,
                dropout_keep_prob: 1.0
            })
            all_predictions = np.concatenate(
                [all_predictions, batch_predictions])

# 如果提供了标签则打印正确率
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print('\nTotal number of test examples: {}'.format(len(y_test)))
    print('Accuracy: {:g}'.format(correct_predictions / float(len(y_test))))

# 保存评估为csv
predictions_human_readable = np.column_stack((np.array(x_raw),
                                              all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, '..', 'prediction.csv')
print('Saving evaluation to {0}'.format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)