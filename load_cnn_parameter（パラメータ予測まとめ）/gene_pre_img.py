import glob
import cv2
import numpy as np
import random


def gen_images(num_image, row,col, size_min, size_max, ang_max,tate,yoko):

    # パラメーター一覧
    #  画像一枚につき何枚
    #num_image
    #  生成する画像の一辺のサイズ（正方形を想定）
    #image_size
    # 以下の範囲内でランダムに画像を拡大、縮小。
    #  画像を縮小するときの最小%
    #size_min
    #  画像を拡大するときの最大%
    #size_max
    # 画像を回転させる時の最大角度
    #ang_max
    #　縦方向への最大移動距離　ピクセルで指定
    #tate = 50
    #　横方向への最大移動距離　ピクセルで指定
    #yoko = 50

    #リサイズする縦横　ピクセルで指定
    #縦：row = 280
    #横：col = 180

    size = (row,col)
    image_list = glob.glob("pre_image/*")

    label_list=[]
    images = []
    for i in range(num_image):

        #img = cv2.imread('image/menkyo2.jpg')
        for img in image_list:
            img = cv2.imread(img)
            img = cv2.resize(img, size)

            row ,col ,ch = img.shape
            #画像のサイズ
            scale = random.randint(size_min,size_max)
            scale = scale / 100

            #　平行移動
            
            dy =random.randint(0,tate)
            dx =random.randint(0,yoko)

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
            #cv2.imwrite('new_image/'+str(i)+'.jpg',dst2)
            
    images  = np.array(images)
    label_list  = np.array(label_list)
    
    return images, label_list
    