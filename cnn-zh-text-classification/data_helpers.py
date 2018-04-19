import numpy as np
import re
import os


def load_data_and_labels(data_files):
    """
    1. 加载所有数据和标签
    2. 可以进行多分类，每个类别的数据单独放在一个文件中
    3. 保存处理后的数据
    """
    data_files = data_files.split(',')
    num_data_file = len(data_files)
    assert num_data_file > 1
    x_text = []
    y = []
    for i, data_file in enumerate(data_files):
        # 将数据放在一起
        data = read_and_clean_zh_file(data_file, True)
        x_text += data
        # 形成数据对应的标签
        label = [0] * num_data_file
        label[i] = 1
        labels = [label for _ in data]
        y += labels
    return [x_text, np.array(y)]


def read_and_clean_zh_file(input_file, output_cleaned_file=False):
    """
    1. 读取中文文件并清洗句子
    2. 可以将清洗后的结果保存到文件
    3. 如果已经存在经过清洗的数据文件则直接加载
    """
    data_file_path, file_name = os.path.split(input_file)
    output_file = os.path.join(data_file_path, 'cleaned_' + file_name)
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
    """
    1. 将除汉字外的字符转为一个空格
    2. 将连续的多个空格转为一个空格
    3. 除去句子前后的空格字符
    """
    string = re.sub(r'[^\u4e00-\u9fff]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip()


def seperate_line(line):
    """
    将句子中的每个字用空格分隔开
    """
    return ''.join([word + ' ' for word in line])


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    生成一个batch迭代器
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]


if __name__ == '__main__':
    data_files = './data/maildata/spam_5000.utf8,./data/maildata/ham_5000.utf8'
    x_text, y = load_data_and_labels(data_files)
    print(len(y))
