import cv2
import numpy as np

# 最大长度
maxLength = 700
minArea = 2000


# w,h-原图的尺寸
# wMax,hMax-图像的最大宽度和最大高度
def zoom(w, h, wMax, hMax):
    # 图像宽度的比例
    widthScale = 1.0 * wMax / w
    # 图像高度的比例
    heightScale = 1.0 * hMax / h
    # 比较宽度比例和高度比例，选择最小的比例
    scale = min(widthScale, heightScale)
    # 重新定位高度和宽度
    resizeWidth = int(w * scale)
    resizeHeight = int(h * scale)
    # 返回新的高度和宽度
    return resizeWidth, resizeHeight

# 点的限制
def pointLimit(point, maxWidth, maxHeight):
    if point[0] < 0:
        point[0] = 0
    if point[0] > maxWidth:
        point[0] = maxWidth
    if point[1] < 0:
        point[1] = 0
    if point[1] > maxHeight:
        point[1] = maxHeight

# 图片预处理
def pre(img):
    # 高斯模糊降低噪声
    img = cv2.GaussianBlur(img, (3, 3), 0)
    imgGary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("imgGary", imgGary)
    # 开运算强化对比度
    kernel = np.ones((20, 20), np.uint8)
    imgOpen = cv2.morphologyEx(imgGary, cv2.MORPH_OPEN, kernel)
    cv2.imshow("imgOpen", imgOpen)
    # 加权强化对比度
    imgOpenWeight = cv2.addWeighted(imgGary, 1, imgOpen, -1, 0)
    cv2.imshow("imgOpenWeight", imgOpenWeight)
    # 二值化操作找到物体轮廓
    ret, imgBin = cv2.threshold(imgOpenWeight, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow("imgBin", imgBin)
    # Canny边缘检测找到物体轮廓
    imgEdge = cv2.Canny(imgBin, 100, 200)
    cv2.imshow("imgEdge", imgEdge)
    # 先闭后开运算找到整块的矩形
    kernel = np.ones((10, 19), np.uint8)
    img_close = cv2.morphologyEx(imgEdge, cv2.MORPH_CLOSE, kernel)
    imgEdge = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel)
    # 由于部分图像得到的轮廓边缘不整齐，因此再进行一次膨胀操作
    img2 = cv2.dilate(imgEdge, np.ones(shape=[5, 5], dtype=np.uint8), iterations=3)
    return img2


# 寻找包络,删除一些物理尺寸不满足的包络
def find_contours(img2):
    contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > minArea]
    global carPlateList
    carPlateList = []
    imgDark = np.zeros((img2.shape), dtype=img2.dtype)
    for index, contour in enumerate(contours):
        # [中心(x,y), (宽,高), 旋转角度]
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w < h:
            w, h = h, w
        scale = w / h
        if scale > 2 and scale < 4:
            color = (0, 245, 255)
            carPlateList.append(rect)
            # 第一个参数表示目标图像
            # 第二个参数表示输入的轮廓组
            # 第三个参数表示画第几个轮廓
            # 第四个参数为轮廓的颜色
            # 第五个参数为轮廓的线宽
            # 第六个参数为轮廓的线性（为8连通的）
            cv2.drawContours(imgDark, contours, index, color, 1, 8)

            # 峰值坐标
            box = cv2.boxPoints(rect)
            # 将box转换成整数值
            box = np.int0(box)
            cv2.drawContours(imgDark, [box], 0, (239, 0, 0), 1)
            print("Vehicle number: ", len(carPlateList))
            return imgDark

# 重映射(仿射变换)，矫正得到的车牌矩形
def correct(img):
    global carPlateList
    global imgPlatList
    imgPlatList = []
    for index, carPlat in enumerate(carPlateList):
        if carPlat[2] > -1 and carPlat[2] < 1:
            angle = 1
        else:
            angle = carPlat[2]
        carPlat = (carPlat[0], (carPlat[1][0] + 5, carPlat[1][1] + 5), angle)
        box = cv2.boxPoints(carPlat)

        # 哪个点是左/右/上/下
        w, h = carPlat[1][0], carPlat[1][1]
        if w > h:
            LT = box[1]
            LB = box[0]
            RT = box[2]
            RB = box[3]
        else:
            LT = box[2]
            LB = box[1]
            RT = box[3]
            RB = box[0]
        for point in [LT, LB, RT, RB]:
            pointLimit(point, imgWidth, imgHeight)
        # 用warpAffine()函数对图像进行旋转摆正
        newLB = [LT[0], LB[1]]
        newRB = [RB[0], LB[1]]
        oldTriangle = np.float32([LT, LB, RB])
        newTriangle = np.float32([LT, newLB, newRB])
        # 得到变换矩阵
        warpMat = cv2.getAffineTransform(oldTriangle, newTriangle)
        # 进行仿射变换
        imgAffine = cv2.warpAffine(img, warpMat, (imgWidth, imgHeight))
        cv2.imshow("imgAffine"+str(index), imgAffine)
        print("Index: ", index)
        imgPlat = imgAffine[int(LT[1]):int(newLB[1]), int(newLB[0]):int(newRB[0])]
        imgPlatList.append(imgPlat)
        cv2.imshow("imgPlat"+str(index), imgPlat)
# 图片缩放到固定的大小
imgOri = cv2.imread(
    'D:/License-plate-recognition/01-90_89-279&506_467&562-464&564_281&560_284&498_467&502-0_0_32_8_26_26_30-127-34.jpg')
cv2.imshow('original', imgOri)
img = np.copy(imgOri)
h, w = img.shape[:2]
imgWidth, imgHeight = zoom(w, h, maxLength, maxLength)
print(w, h, imgWidth, imgHeight)
# (imgWidth, imgHeight)-输出图像尺寸
# INTER_AREA-使用像素区域关系进行重采样(首选:它会产生无云纹理的结果)
img = cv2.resize(img, (imgWidth, imgHeight), interpolation=cv2.INTER_AREA)
cv2.imshow("imgResize", img)
img2 = pre(img)
cv2.imshow("pre", img2)
imgDark = find_contours(img2)
cv2.imshow("imgGaryContour", imgDark)
correct(img)
cv2.waitKey(0)
