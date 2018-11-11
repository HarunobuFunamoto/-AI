#  画像1つに対して、パラメーター４つのラベルをつける

import glob
import os
import cv2
import numpy as np
import random
from sklearn import cross_validation
from PIL import Image

def generate_images(num_image, image_size, size_min, size_max, ang_max):

    # パラメーター一覧
    #  生成する画像の数
    num_image
    #  生成する画像の一辺のサイズ（正方形を想定）
    image_size
    # 以下の範囲内でランダムに画像を拡大、縮小。
    #  画像を縮小するときの最小%
    size_min
    #  画像を拡大するときの最大%
    size_max
    # 画像を回転させる時の最大角度
    ang_max

    size = (280,180)
    image_list = glob.glob("image/*")

    label_list=[]
    images = []
    for i in range(num_image):

        #img = cv2.imread('image/menkyo2.jpg')
        img = cv2.imread(random.choice(image_list))
        img = cv2.resize(img, size)

        row ,col ,ch = img.shape
        #画像のサイズ
        scale = random.randint(size_min,size_max)
        scale = scale / 100

        #　平行移動
        move = image_size/ 2
        dy =random.randint(0,move)
        dx =random.randint(0,move)

        # 回転角度
        ang = random.randint(0,ang_max)

        # これらのパラメーターをラベルとして格納
        para = [scale,dy,dx,ang]
        label_list.append(para)


       #  回転、拡大縮小、平行移動を実施
        M = cv2.getRotationMatrix2D( (col/2 , row/2), ang,scale)
        dst = cv2.warpAffine(img,M,(col,row))
        M1 = np.float64([[1,0,dx],[0,1,dy]])
        dst2 = cv2.warpAffine(dst,M1,(col,row))

        images.append(dst2)


    data_train, data_test, label_train, label_test = cross_validation.train_test_split(images, label_list)
    xy = (data_train, data_test, label_train, label_test)
    np.save('data/train_test.npy', xy)
    num_train = len(data_train)
    num_test = len(data_test)
    print('作った画像の数：',num_image)
    print('学習用の画像の数：',num_train)
    print('テスト用の画像の数：',num_test)

