# -*- coding: utf-8 -*-
# @Author  : 谭媚
# @Time    : 2022/6/30 20:47
# @File    : net.py
# @Software: PyCharm
import os  # 操作系统接口模块
import cv2  # 引用opencv库
import numpy as np  # 主要用于数组与矩阵计算的数学库
import keras  # keras神经网络框架
from keras import layers, models

#####卷积神经网络#####

# 训练集
def train():
    path_train = 'D:/License-plate-recognition/images/train/'  # 训练集的路径
    index = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
             "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
             "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
             "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
             "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
             "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
             "W": 61, "X": 62, "Y": 63, "Z": 64}
    imageName = os.listdir(path_train)  # 打开训练集文件
    n = len(imageName)  # 训练集中图片个数
    x1, y1 = [], []  # 两个数组矩阵包含所有训练集
    i = 0
    while i < n:
        # 用cv2.imdecode()函数读取数据把数据转换成图像格式打开图片,-1即打开源文件
        image = cv2.imdecode(np.fromfile(
            path_train+imageName[i], dtype="uint8"), -1)
        # 寻找每个字符对应的编号作为标签
        k = []
        for char in imageName[i][0:7]:
            k.append(index[char])
        x1.append(image)  # 把图片送入x1矩阵
        y1.append(k)  # 把k送入y1矩阵
        i = i+1
    print("已经读了%d张照片" % i)
    # 将训练集转化为numpy标准数组格式便于运行
    x1 = np.array(x1)
    y1 = [np.array(y1)[:, i] for i in range(7)]
    # 卷积神经网络
    input_layer = keras.layers.Input((80, 240, 3))
    x = input_layer
    #一组卷积层和七个全连接层
    # 四个卷积层，四个池化层
    x = keras.layers.Conv2D(16, (3, 3), activation='relu')(x)  # 卷积核个数，卷积核大小
    x = keras.layers.MaxPooling2D((2, 2), padding='same', strides=(2, 2))(x)
    for i in range(3):
        x = keras.layers.Conv2D(16*2**i, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(
            (2, 2), padding='same', strides=(2, 2))(x)
    # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
    x = keras.layers.Flatten()(x)
    # 防止过拟合，0.5控制断开神经元的比例
    x = keras.layers.Dropout(0.5)(x)
    # 七个全连接层，每个全连接层对应65个字符
    output = [keras.layers.Dense(
        65, activation='softmax', name='c%d' % (i+1))(x) for i in range(7)]
    model = models.Model(inputs=input_layer, outputs=output)
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 模型训练
    print("开始训练")
    model.fit(x1, y1, epochs=20)
    model.save('cnn_net.h5')
    print('cnn_net.h5保存成功')


# 预测函数用来输出识别结果
def cnn_predict(cnn, Lic_img):
    chars = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
                  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    img = Lic_img
    Lic_pred = []
    lic_pred = cnn.predict(img.reshape(1, 80, 240, 3))  # 预测形状应为(1,80,240,3)
    lic_pred = np.array(lic_pred).reshape(7, 65)  # 列表转为ndarray，形状为(7,65)
    if len(lic_pred[lic_pred >= 0.8]) >= 2:  # 统计其中预测概率值大于80%以上的个数，大于等于4个以上认为识别率高，识别成功
        char = ''
        for arg in np.argmax(lic_pred, axis=1):  # 取每行中概率值最大的arg,将其转为字符
            char += chars[arg]
        char = char[0:2] + '·' + char[2:]
        Lic_pred.append((img, char))  # 将车牌和识别结果一并存入Lic_pred
    return Lic_pred

