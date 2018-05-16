import os
import tensorflow as tf
import numpy as np
import glob
import random
from tensorflow.python.platform import gfile


INPUT_DATA = 'a_test'
BATCH = 100


# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型中代表瓶颈层结果的张量名称。
# 在谷歌提出的Inception-v3模型中，这个张量名称就是'pool_3/_reshape:0'。
# 在训练模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称。
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的谷歌训练好的Inception-v3模型文件目录
MODEL_DIR = 'inception_dec_2015'

# 下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 这个函数从数据文件夹中读取所有的图片列表并按训练、验证、测试数据分开。
# testing_percentage和validation_percentage参数指定了测试数据集和验证数据集的大小。
def create_image_lists(testing_percentage, validation_percentage):
    # 得到的所有图片都存在result这个字典(dictionary)里。
    # 这个字典的key为类别的名称，value也是一个字典，字典里存储了所有的图片名称。
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前目录，不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件。
        extensions = ['jpg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.'+ extension)
            file_list.extend(glob.glob(file_glob))  #獲取文件路徑
        if not file_list:
            continue

        # 通过目录名获取类别的名称。
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机将数据分到训练数据集、测试数据集和验证数据集。  #無法保證數據的數量
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据放入结果字典。
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
            }
    # 返回整理好的所有数据
    return result

# 这个函数获取一张图片经过Inception-v3模型处理之后的特征向量。
# 这个函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件。
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径。
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottlenect_path(image_lists, label_name, index, category)
    # 如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，并将计算的结果存入文件。
    if not os.path.exists(bottleneck_path):
        # 获取原始的图片路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片内容。
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # print(len(image_data))
        # 由于输入的图片大小不一致，此处得到的image_data大小也不一致（已验证），但却都能通过加载的inception-v3模型生成一个2048的特征向量。具体原理不详。
        # 通过Inception-v3模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中获取图片相应的特征向量。
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回得到的特征向量
    return bottleneck_values

# 这个函数随机获取一个batch的图片作为训练数据。
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个类别和图片的编号加入当前的训练数据。
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, category,
                                              jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

# 这个函数获取全部的测试数据。在最终测试的时候需要在所有的测试数据上计算正确率。
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有的类别和每个类别中的测试图片。
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据的列表。
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                                                  jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype = np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
    graph_def = tf.GraphDef()
# 加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量。
bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        image_lists = create_image_lists(10, 10)
        n_classes = len(image_lists.keys())

        # 每次获取一个batch的训练数据
        train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
            sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
        # 在验证集上测试正确率。

        validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
            sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)

        # 在最后的测试数据上测试正确率
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes,
                                                                   jpeg_data_tensor, bottleneck_tensor)


    result = create_image_lists(10, 10)
    print(result.values())
