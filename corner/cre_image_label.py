import cv2
import numpy as np
import random
from sklearn import cross_validation


#  以下は画像を読み込んで、回転、拡大縮小、平行移動する処理
def image_point(num_image, size_min, ang_max,tate,yoko, x1,y1,x2,y2,x3,y3,x4,y4):
    # パラメーター一覧
    #  生成する画像の数
    #num_image = 1
    # 以下の範囲内でランダムに画像を拡大、縮小、移動。
    #  画像を縮小するときの最小%
    #size_min = 70
    #  画像を拡大するときの最大% 極力はみ出さないようにするために固定
    size_max = 100
    # 画像を回転させる時の最大角度
    #ang_max = 360
    # 平行移動させる最大の距離（ピクセル指定）
    #tate
    #yoko

    # メモした画像の頂点の座標を記入する　ピクセルで記入
    # x1　左上のx座標
    # y1　左上のy座標
    # x2　左下のx座標
    # y2　左下のy座標
    # x3　右上のx座標
    # y3　右上のy座標
    # x4　右下のx座標
    # y4　右下のy座標



    # 左上
    p1 = np.array([x1,y1,1])
    # 左下
    p2 = np.array([x2,y2,1])
    # 右上
    p3 = np.array([x3,y3,1])
    # 右下
    p4 = np.array([x4,y4,1])


    zahyou_list = []
    images = []
    for i in range(num_image):
        zahyou = []
        point1=[]
        point2=[]
        point3=[]
        point4=[]
        
        
        # リサイズした時に座標をメモしてここのp１p２に書き込んでから、このプラグラム起動！！リサイズした画像をロード！！
        img = cv2.imread('new_image/menkyo.jpg')
        
        
        row ,col ,ch = img.shape
        #画像のサイズ
        scale = random.randint(size_min,size_max)
        scale = scale / 100
        
        #　平行移動
        dy =random.randint(0,tate)
        dx =random.randint(0,yoko)
        
        # 回転角度
        ang = random.randint(0,ang_max)
        

    #  回転、拡大縮小、平行移動を実施
        M = cv2.getRotationMatrix2D( (col/2 , row/2), ang,scale)
        dst = cv2.warpAffine(img,M,(col,row))
        M1 = np.float64([[1,0,dx],[0,1,dy]])
        dst2 = cv2.warpAffine(dst,M1,(col,row))
        
        
        
        images.append(dst2)
        
        
        #images = np.array(images)
        #images = images.astype("float") / 255
        
        
        # 座標の回転
        p1 = np.dot(M,p1)
        p2 = np.dot(M,p2)
        p3 = np.dot(M,p3)
        p4 = np.dot(M,p4)
        
        p1 = np.append( p1,'1')
        p2 = np.append( p2,'1')
        p3 = np.append( p3,'1')
        p4 = np.append( p4,'1')
        
        p1 = p1.astype('float64')
        p2 = p2.astype('float64')
        p3 = p3.astype('float64')
        p4 = p4.astype('float64')
        
        # さらに平行移動
        after_p1 = np.dot(M1,p1)
        after_p2 = np.dot(M1,p2)
        after_p3 = np.dot(M1,p3)
        after_p4 = np.dot(M1,p4)
        
        
        # 回転、移動させた画像を%に変換！！
        xper_p1 = after_p1[0] / col
        yper_p1 = after_p1[1] / row
        point1.append(xper_p1)
        point1.append(yper_p1)
        
        xper_p2 =after_p2[0] / col
        yper_p2 =after_p2[1] / row
        point2.append(xper_p2)
        point2.append(yper_p2)
        
        xper_p3 =after_p3[0] / col
        yper_p3 =after_p3[1] / row
        point3.append(xper_p3)
        point3.append(yper_p3)

        xper_p4 =after_p4[0] / col
        yper_p4 =after_p4[1] / row
        point4.append(xper_p4)
        point4.append(yper_p4)
        
        
        zahyou.append(point1)
        zahyou.append(point2)
        zahyou.append(point3)
        zahyou.append(point4)
        
        zahyou_list.append(zahyou)
        
        
    image_train, image_test, label_train, label_test = cross_validation.train_test_split(images, zahyou_list)

    return image_train, image_test, label_train, label_test

