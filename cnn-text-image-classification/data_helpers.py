import os, re, glob, os, random
import numpy as np
from skimage import io, transform
from tensorflow.contrib import learn

# 加载经过处理后的数据，划分为训练和测试
# ==================================================


def load_train_valid_data(text_directory, image_directory,
                          validation_percentage, image_size):
    print('Loading data...')
    texts, images, y = load_texts_images_labels(text_directory,
                                                image_directory)
    print('Total text-images: {:d}'.format(len(y)))

    max_document_length = max([len(x.split(' ')) for x in texts])
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)
    texts = np.array(list(vocab_processor.fit_transform(texts)))
    print('Vocabulary Size: {:d}'.format(len(vocab_processor.vocabulary_)))
    print('')

    np.random.seed(10)
    shuffle_indexes = np.random.permutation(np.arange(len(y)))
    texts_shuffled = texts[shuffle_indexes]
    images_shuffled = images[shuffle_indexes]
    y_shuffled = y[shuffle_indexes]

    validation_indexes = -1 * int(validation_percentage * len(y))
    if validation_indexes == 0:
        validation_indexes = -1
    text_train, text_valid = texts_shuffled[:
                                            validation_indexes], texts_shuffled[
                                                validation_indexes:]
    image_train, image_valid = images_shuffled[:
                                               validation_indexes], images_shuffled[
                                                   validation_indexes:]
    y_train, y_valid = y_shuffled[:validation_indexes], y_shuffled[
        validation_indexes:]
    print('train/valid split: {:d}/{:d}'.format(len(y_train), len(y_valid)))
    print('')

    return text_train, text_valid, read_images(
        image_train, image_size), read_images(
            image_valid, image_size), y_train, y_valid, vocab_processor


def load_texts_images_labels(text_directory, image_directory):
    texts = []
    images = []
    y = []
    text_files = [sub[2] for sub in os.walk(text_directory)][0]
    image_dirs = [sub[0] for sub in os.walk(image_directory)]
    for i, text_file in enumerate(text_files):
        text_data = read_and_clean_zh_file(text_directory + text_file, True)
        texts += text_data
        image_data = load_images_path(image_directory,
                                      text_file.split('.')[0], len(text_data))
        images += image_data
        label = [0] * len(text_files)
        label[i] = 1
        labels = [label for _ in text_data]
        y += labels

    return np.array(texts), np.array(images), np.array(y)


# 读取文本并进行处理
# ==================================================


def read_and_clean_zh_file(input_file, output_cleaned_file=False):
    data_file_path, file_name = os.path.split(input_file)
    output_file = os.path.join(data_file_path, 'cleaned', file_name)
    if os.path.exists(output_file):
        lines = list(open(output_file, 'r').readlines())
        lines = [line.strip() for line in lines]
    else:
        lines = list(open(input_file, 'r').readlines())
        lines = [clean_str(seperate_line(line)) for line in lines]
        if output_cleaned_file:
            with open(output_file, 'w') as f:
                for line in lines:
                    f.write(line + '\n')
    return lines


def clean_str(string):
    string = re.sub(r'[^\u4e00-\u9fff]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip()


def seperate_line(line):
    return ''.join([word + ' ' for word in line])


# 读取图像并进行处理
# ==================================================


def load_images_path(image_directory, input_file, num_images):
    images = []
    sub_dirs = [sub[0] for sub in os.walk(image_directory)]
    for i, sub_dir in enumerate(sub_dirs):
        if i == 0:
            continue
        dir_name = os.path.basename(sub_dir)
        if dir_name != input_file:
            continue
        exts = ['jpg', 'jpeg', 'JPG', 'JPEG']
        for ext in exts:
            image_glob = os.path.join(image_directory, dir_name, '*.' + ext)
            images.extend(glob.glob(image_glob))
    images_expand = []
    for _ in range(num_images):
        i = random.randint(0, len(images) - 1)
        images_expand.append(images[i])

    return images_expand


def read_images(images, image_size):
    result = []
    for i, im in enumerate(images):
        print(str(i) + ' ' + im)
        img = io.imread(im)
        img = transform.resize(img, (image_size, image_size))
        result.append(img)
    return result


# 生成批处理数据
# ==================================================


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


if __name__ == '__main__':
    text_directory = './data/text/'
    image_directory = './data/image/'
    load_train_valid_data(text_directory, image_directory, 0.1, 28)
