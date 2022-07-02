import cv2 as cv
import numpy as np
# -*- coding: utf-8 -*-
# @Author  : 谭媚
# @Time    : 2022/6/30 20:50
# @File    : cut.py
# @Software: PyCharm

# 创建一个调用函数，用来得出分割字符的终点横坐标
def find_end(start_, black, black_max):
    width = 136  # 前面归一化后的宽度值
    segmentation_spacing = 0.95  # 判断阈值，可修改
    end_ = start_ + 1
    for g in range(start_ + 1, width - 1):
        if black[g] > segmentation_spacing * black_max:
            end_ = g
            break
        if g == 134:
            end_ = g
            break
    return end_


# 创建一个二级调用函数
def find_endl(start_1, black, black_max):
    width = 136  # 前面归一化后的宽度值
    segmentation_spacing_1 = 0.75  # 二级判断阈值
    end_1 = start_1 + 1
    for r in range(start_1 + 1, width - 1):
        if black[r] > segmentation_spacing_1 * black_max:
            end_1 = r
            break
    return end_1


# 创建一个函数，期望实现将切割后的图像移到中央
def expand(img):
    img_1 = img
    height = img_1.shape[0]
    width = img_1.shape[1]
    img_2 = np.zeros((36, 36), np.uint8)
    for i in range(0, 36):
        for j in range(0, 36):
            if j < 9:
                img_2[i, j] = 0
            elif j < width + 9:
                img_2[i, j] = img_1[i, j - 9]
            else:
                img_2[i, j] = 0
    # cv.imshow('df',img_2)
    # cv.waitKey(0)
    return img_2


# 读取图像,并显示图像，将图像归一化，宽为136，高为36
def cut(img):
    img_resize = cv.resize(img, (136, 36), interpolation=cv.INTER_AREA)  # 图像归一化
    # 灰度化+二值化，这一步主要是尝试
    img_gray_1 = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
    ret1, img_thre_1 = cv.threshold(img_gray_1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    img_gaus = cv.GaussianBlur(img_resize, (3, 5), 0)  # 高斯模糊
    img_gray = cv.cvtColor(img_gaus, cv.COLOR_BGR2GRAY)
    ret, img_thre = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # 正二值化，采用OTSU最优值

    # 除去长横线的噪声干扰
    height = img_thre.shape[0]  # 读取行数36
    width = img_thre.shape[1]  # 读取列数136
    sum_1 = 0  # 发生跳变
    sum_2 = 0  # 一长段未发生跳变
    sum_3 = []  # 记录每一行像素的跳变次数
    sum_4 = []
    # 记录跳变次数
    for a in range(height):
        s1 = 0
        for b in range(width - 1):
            s2 = img_thre[a, b]
            s3 = img_thre[a, b + 1]
            if s2 != s3:
                s1 = s1 + 1
        sum_3.append(s1)
    print(sum_3)
    # 将干扰的噪点的像素值置0
    img_threC = img_thre
    for i in range(height):
        sum_1 = 0
        sum_2 = 0
        for j in range(width - 1):
            s4 = img_thre[i, j]
            s5 = img_thre[i, j + 1]
            if s4 != s5:  # 判断像素是否发生跳变
                sum_1 = sum_1 + 1
                sum_2 = 0
            else:
                sum_2 = sum_2 + 1
            if sum_2 != 0:
                if int(width / sum_2) < 5:  # 未跳变的线段长超过了总长的1/5
                    sum_1 = 0
                    for c in range(width - 1):  # 将干扰行像素全部置零
                        img_threC.itemset((i, c), 0)
                break
    for k in range(height):  # 存在跳变次数小于7的行数消除该行像素值
        if sum_3[k] < 10:
            for x in range(width):
                img_threC.itemset((k, x), 0)

    # 记录消除后的跳变次数
    for d in range(height):
        s6 = 0
        for e in range(width - 1):
            s7 = img_threC[d, e]
            s8 = img_threC[d, e + 1]
            if s7 != s8:
                s6 = s6 + 1
        sum_4.append(s6)
    print(sum_4)

    # 仍然两幅图片对比，相同置一
    img_x = np.zeros(img_thre.shape, np.uint8)  # 重新拷贝图片
    height_x = img_resize.shape[0]  # 行数
    width_x = img_resize.shape[1]  # 列数
    for i in range(height_x):
        for j in range(width_x):
            h_x = img_threC[i][j]
            s_x = img_thre_1[i][j]
            if h_x == 255 and s_x == 255:
                img_x[i][j] = 255
            else:
                img_x[i][j] = 0
    # cv.imshow('threshold',img_x)
    # cv.waitKey(0)

    # 对消除噪声后的图片进行闭操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 4))  # 设置开闭操作卷积核大小3*4  后一个应该为宽度
    img_close = cv.morphologyEx(img_threC, cv.MORPH_CLOSE, kernel)  # 闭操作

    # 采用投影法，获得需裁剪的字符的横坐标的开始和结束
    segmentation_spacing = 0.95  # 判断阈值，可修改
    segmentation_spacing_1 = 0.85  # 二级判断阈值，用来解决一些字符粘连的问题
    white = []  # 记录每一列的白色像素总和
    black = []  # 记录每一列的黑色像素总和
    white_max = 0  # 仅保存每列，取列中白色最多的像素总数
    black_max = 0  # 仅保存每列，取列中黑色最多的像素总数
    # 循环计算每一列的黑白色像素总和
    for q in range(width):
        w_count = 0  # 这一列白色总数
        b_count = 0  # 这一列黑色总数
        for w in range(height):
            h = img_close[w, q]
            if h == 0:
                b_count = b_count + 1
            else:
                w_count = w_count + 1
        white_max = max(white_max, w_count)
        black_max = max(black_max, b_count)
        white.append(w_count)
        black.append(b_count)

    # 分割字符
    n = 0
    start = 0
    end = 1
    a = 1
    while n < width - 1:
        n = n + 1
        if white[n] > (1 - segmentation_spacing) * white_max:
            start = n
            end = find_end(start, black, black_max)
            if end - start > 3 and end - start <= 18:
                if end - start < 8:
                    print(start - 4, end + 4)
                    img_cut = img_x[0:height, start - 4:end + 4]
                    img_cut1 = expand(img_cut)
                    img_final = cv.resize(img_cut1, (32, 32), interpolation=cv.INTER_AREA)  # 将裁剪后的字符归一化32*32大小
                    cv.imwrite("D:\License-plate-recognition\\test222\\%s.bmp" % a, img_final)
                    print("保存:%s" % a)
                    a = a + 1
                else:
                    print(start, end)
                    img_cut = img_x[0:height, start:end]
                    img_cut1 = expand(img_cut)
                    img_final = cv.resize(img_cut1, (32, 32), interpolation=cv.INTER_AREA)
                    cv.imwrite("D:\License-plate-recognition\\test222\\%s.bmp" % a, img_final)
                    print("保存:%s" % a)
                    a = a + 1
                # cv.imshow("cutchar",img_final)
                # cv.waitKey(0)
            if end - start > 18:
                end = find_endl(start, black, black_max)
                print(start, end)
                img_cut = img_x[0:height, start:end]
                img_cut1 = expand(img_cut)
                img_final = cv.resize(img_cut1, (32, 32), interpolation=cv.INTER_AREA)
                cv.imwrite("D:\License-plate-recognition\\test222\\%s.bmp" % a, img_final)
                print("保存:%s" % a)
                a = a + 1
                # cv.imshow("cutchar",img_final)
                # cv.waitKey(0)
            n = end


if __name__ == "__main__":
    img = cv.imread("D:\License-plate-recognition\yolov5/cut_plate.jpg")  # 读取图像
    # cv.imshow("image",img)                           #显示图像
    # cv.waitKey(0)
    # imgp = pl.location(img)
    cut(img)