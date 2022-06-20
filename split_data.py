import os
import random
from shutil import copy2
trainfiles = os.listdir(r"D:\License-plate-recognition\base")  #（图片文件夹）
num_train = len(trainfiles)
print("num_train: " + str(num_train) )
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)  # 打乱顺序
num = 0
trainDir = r"D:\License-plate-recognition\image\train"
validDir = r"D:\License-plate-recognition\image\val"
for i in index_list:
    fileName = os.path.join(r"D:\License-plate-recognition\base", trainfiles[i])  #（图片文件夹）+图片名=图片地址
    if num < num_train*0.7:
        print(str(fileName))
        copy2(fileName, trainDir)
    elif num < num_train*0.9:
        print(str(fileName))
        copy2(fileName, validDir)
    num += 1
