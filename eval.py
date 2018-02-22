import tensorflow as tf
import numpy as np
import data_helpers
from image_cnn import ImageCNN

# 参数
# ==================================================

flags = tf.flags

# 数据参数
flags.DEFINE_string('data_directory', './data/flower_photos/',
                    'Image data directory path')

# 评估参数
flags.DEFINE_integer('batch_size', 100, 'Batch Size')
flags.DEFINE_integer('image_size', 28, 'Image size for resizing')
flags.DEFINE_string('checkpoint_dir', './runs/1519316254/checkpoints',
                    'Checkpoint directory from training run')
flags.DEFINE_boolean('eval_train', False, 'Evaluate on all training data')

# 打印参数
FLAGS = flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('')

# 加载训练数据或者自己提供测试数据
if FLAGS.eval_train:
    x, _, y, _ = data_helpers.load_train_valid_data(FLAGS.data_directory, 0,
                                                    FLAGS.image_size)
    y_test = np.argmax(y, axis=1)
else:
    test_image_path = './data/test_images/'
    x, _, y, _ = data_helpers.load_train_valid_data(test_image_path, 0,
                                                    FLAGS.image_size)

# 评估
# ==================================================

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
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
            list(x), FLAGS.batch_size, 1, shuffle=False)

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