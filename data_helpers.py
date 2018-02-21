import numpy as np
import tensorflow as tf
import glob, os
from skimage import io, transform


def load_train_valid_data(data_directory, validation_percentage, image_size):
    # 加载数据
    print('Loading data...')
    x, y = load_images_and_labels(data_directory)
    print('Total images: {:d}'.format(len(y)))

    # 随机混淆数据
    np.random.seed(10)
    shuffle_indexes = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indexes]
    y_shuffled = y[shuffle_indexes]

    # 划分train/test集
    validation_indexes = -1 * int(validation_percentage * len(y))
    if validation_indexes == 0:
        validation_indexes = -1
    x_train, x_valid = x_shuffled[:validation_indexes], x_shuffled[
        validation_indexes:]
    y_train, y_valid = y_shuffled[:validation_indexes], y_shuffled[
        validation_indexes:]

    print('train/valid split: {:d}/{:d}'.format(len(x_train), len(x_valid)))
    print('')

    return read_images(x_train, image_size), read_images(
        x_valid, image_size), y_train, y_valid


def load_images_and_labels(data_directory):
    images = []  # 存储所有图片的路径
    labels = []  # 每个图片对应的标签
    sub_dirs = [sub[0] for sub in os.walk(data_directory)]  # 获取所有子目录

    # 获取每个子目录的图片
    for i, sub_dir in enumerate(sub_dirs):
        # 跳过当前根目录
        if i == 0:
            continue

        # 获取当前子目录下的有效图片名称
        dir_images = []  # 存储当前目录下的图片用于添加标签
        exts = ['jpg', 'jpeg', 'JPG', 'JPEG']
        dir_name = os.path.basename(sub_dir)  # 获取路径的最后一个目录名
        for ext in exts:
            image_glob = os.path.join(data_directory, dir_name, '*.' + ext)
            dir_images.extend(glob.glob(image_glob))
        images.extend(dir_images)

        # 添加对应标签
        label = [0] * (len(sub_dirs) - 1)
        label[i - 1] = 1
        labels.extend([label for _ in dir_images])

    return np.array(images), np.array(labels)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for _ in range(num_epochs):
        if shuffle:
            shuffle_indexes = np.random.permutation(np.arange(data_size))
            data = data[shuffle_indexes]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min(start_index + batch_size, data_size)
            yield data[start_index:end_index]


def read_images(images, image_size):
    result = []
    print('\nReading images:')
    for i, im in enumerate(images):
        # print(str(i) + ' ' + im)
        img = io.imread(im)
        img = transform.resize(img, (image_size, image_size))
        result.append(img)
    return result


if __name__ == '__main__':
    data_directory = './data/flower_photos/'
    load_train_valid_data(data_directory, 0.1)
