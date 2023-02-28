from flask import Flask,render_template
from flask import jsonify
import random

import matplotlib.pyplot as plt
# this module is used to draw a picture
# usually connected to 'numpy' module
# its usage just like matlab

import numpy as np
# this module is used to calculate or transform arrays and lists

from tensorflow.keras import datasets, layers, models
# tensorflow.keras is a high lever module for python API
# 'from tensorflow.keras import datasets' is used to download datasets
# 'import layers' is used to customize the layers of neural network
# 'import models' is used to customize the whole model of neural network

app = Flask(__name__)

# set number of rounds
BATCH = 100

# the images source
class DataSource(object):
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))

        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0
        self.train_images, self.train_labels = train_images[0:15000], train_labels[0:15000]
        self.test_images, self.test_labels = test_images[0:10000], test_labels[0:10000]


def random_num_with_fix_total(maxvalue, num):
    """生成总和固定的整数序列
    maxvalue: 序列总和
    num：要生成的整数个数"""
    a = random.sample(range(1, maxvalue), k=num - 1)  # 在1~99之间，采集20个数据
    a.append(0)  # 加上数据开头
    a.append(maxvalue)
    a = sorted(a)
    b = [a[count] - a[count - 1] for count in range(1, len(a))]  # 列表推导式，计算列表中每两个数之间的间隔
    return b


class DataSource1(object):
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0
        self.TI, self.TL = train_images[0:15000], train_labels[0:15000]
        self.train_images = np.empty(BATCH, dtype=object)
        self.train_labels = np.empty(BATCH, dtype=object)
        begin = 0
        rand_count = random_num_with_fix_total(15000, BATCH)
        for count in range(100):
            self.train_images[count] = self.TI[begin:(begin + rand_count[count])]
            self.train_labels[count] = self.TL[begin:(begin + rand_count[count])]
            begin = begin + rand_count[count]
        self.test_images, self.test_labels = test_images[0:10000], test_labels[0:10000]


class DataSource2(object):
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0
        self.TI, self.TL = train_images[15000:30000], train_labels[15000:30000]
        self.train_images = np.empty(BATCH, dtype=object)
        self.train_labels = np.empty(BATCH, dtype=object)
        begin = 0
        rand_count = random_num_with_fix_total(15000, BATCH)
        for count in range(100):
            self.train_images[count] = self.TI[begin:(begin + rand_count[count])]
            self.train_labels[count] = self.TL[begin:(begin + rand_count[count])]
            begin = begin + rand_count[count]


class DataSource3(object):
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0
        self.TI, self.TL = train_images[15000:30000], train_labels[15000:30000]
        self.train_images = np.empty(BATCH, dtype=object)
        self.train_labels = np.empty(BATCH, dtype=object)
        begin = 0
        rand_count = random_num_with_fix_total(15000, BATCH)
        for count in range(100):
            self.train_images[count] = self.TI[begin:(begin + rand_count[count])]
            self.train_labels[count] = self.TL[begin:(begin + rand_count[count])]
            begin = begin + rand_count[count]


class DataSource4(object):
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0
        self.TI, self.TL = train_images[15000:30000], train_labels[15000:30000]
        self.train_images = np.empty(BATCH, dtype=object)
        self.train_labels = np.empty(BATCH, dtype=object)
        begin = 0
        rand_count = random_num_with_fix_total(15000, BATCH)
        for count in range(100):
            self.train_images[count] = self.TI[begin:(begin + rand_count[count])]
            self.train_labels[count] = self.TL[begin:(begin + rand_count[count])]
            begin = begin + rand_count[count]


# Define as LeNet
class CNN(object):
    def __init__(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPool2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPool2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        #        model.summary()  #打印网络结构
        self.model = model


# FedAvg Function
def FedAvg():
    weight_CNN_1 = np.load("Client1Weight.npy", allow_pickle=True)
    weight_CNN_2 = np.load("Client2Weight.npy", allow_pickle=True)
    weight_CNN_3 = np.load("Client3Weight.npy", allow_pickle=True)
    weight_CNN_4 = np.load("Client4Weight.npy", allow_pickle=True)
    weight_array = (weight_CNN_1 + weight_CNN_2 + weight_CNN_3 + weight_CNN_4) / 4
    weight_out = np.array(weight_array)
    return weight_out


# EKF Function
def EKF(cnn, weight_in):
    cnn.model.set_weights(weight_in)
    return cnn


# Create Models:LeNet
cnn_sever = CNN()
cnn1 = CNN()
cnn2 = CNN()
cnn3 = CNN()
cnn4 = CNN()
# Prepare Client Data
data_sever = DataSource()
data1 = DataSource1()
data2 = DataSource2()
data3 = DataSource3()
data4 = DataSource4()

# Compile Client and Sever Model
cnn_sever.model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
cnn1.model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
cnn2.model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
cnn3.model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
cnn4.model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
storage_acc = []
weight = cnn_sever.model.get_weights()
np.save("SeverWeight", weight)
# All Clint Train
for i in range(BATCH):
    # Client Model Update(Downloads From Sever)
    weight = np.load("SeverWeight.npy", allow_pickle=True)
    #    cnn1.model.set_weights(weight)
    #    cnn2.model.set_weights(weight)
    #    cnn3.model.set_weights(weight)
    #    cnn4.model.set_weights(weight)
    cnn1 = EKF(cnn1, weight)
    cnn2 = EKF(cnn2, weight)
    cnn3 = EKF(cnn3, weight)
    cnn4 = EKF(cnn4, weight)
    # Client Model Fit
    cnn1.model.fit(data1.train_images[i], data1.train_labels[i], epochs=3)
    cnn2.model.fit(data2.train_images[i], data2.train_labels[i], epochs=3)
    cnn3.model.fit(data3.train_images[i], data3.train_labels[i], epochs=3)
    cnn4.model.fit(data4.train_images[i], data4.train_labels[i], epochs=3)
    # FedAvg

    weight_CNN1 = np.array(cnn1.model.get_weights())
    weight_CNN2 = np.array(cnn2.model.get_weights())
    weight_CNN3 = np.array(cnn3.model.get_weights())
    weight_CNN4 = np.array(cnn4.model.get_weights())


    np.save("Client1Weight", weight_CNN1)
    np.save("Client2Weight", weight_CNN2)
    np.save("Client3Weight", weight_CNN3)
    np.save("Client4Weight", weight_CNN4)
    weight = FedAvg()
    # Uploads to Sever
    cnn_sever.model.set_weights(weight)
    np.save("SeverWeight", weight)
    test_loss, test_acc = cnn_sever.model.evaluate(data_sever.test_images[0:1000], data_sever.test_labels[0:1000])
    print("Sever: 轮次: %d,准确率: %.4f，共测试了%d张图片 " % (i + 1, test_acc, len(data_sever.test_labels)))
    storage_acc = np.append(storage_acc, test_acc)
# Show Acc
x = np.array(range(100))
plt.plot(x, storage_acc)
plt.savefig('./static/acc.png')

@app.route('/')
def index():
    return render_template('index.html', weight = str(weight))

if __name__ == '__main__':
    app.run(host = '0.0.0.0')
