#coding:utf-8

import cv2 
import numpy as np
import matplotlib.pyplot as plt


# 住民票の画像。分割したい画像を入れる。
jumin = 'jumin/jumin05.jpg'

# それぞれのパーツで予測した座標のnpyファイルを指定
# ４分割にする
# 1 2
# 3 4
zahyou1 = 'zahyou4/point00.npy'
zahyou2 = 'zahyou4/point01.npy'
zahyou3 = 'zahyou4/point10.npy'
zahyou4 = 'zahyou4/point11.npy'

# それぞれのパーツで予測した座標のnpyファイルを指定
## 5分割
#    01
# 10 11  12
#    21
# 上の画像01=zahou01 左の画像10=zahou10 右の画像12=zahyou12 下の画像21=zahyou21 中央の画像11=zahyou11
zahyou01 = 'zahyou5/point00.npy'
zahyou10 = 'zahyou5/point01.npy'
zahyou12 = 'zahyou5/point10.npy'
zahyou21 = 'zahyou5/point11.npy'
zahyou11 = 'zahyou5/point12.npy'




def resize_image(img):
    # 正方形にリサイズ→リサイズ後はnew_img
    tmp = img[:, :]
    height, width = img.shape[:2]
    size = height
    limit = width
    #余白の左の黒を作る
    start = int((size - limit) / 2)
    # さっきのstartから幅進んだところを終点とする
    fin = int((size + limit) / 2)
    # 背景の黒画像を作る
    new_img = cv2.resize(np.zeros((1, 1, 3), np.uint8), (size, size))
    if(size == height):
        new_img[:, start:fin] = tmp
    else:
        new_img[start:fin, :] = tmp

     # リサイズ後の一枚だけの画像を保存   
    np.save('image/resize_image.npy',new_img)
    return new_img



# 4分割するクラス
class Devide_image4(object):
    
    def __init__(self,image):
        self.img = cv2.imread(image)

    def devide_image(self):    
        # ４分割にする
        # 1 2
        # 3 4
        new_img = resize_image(self.img)
        h, w = new_img.shape[:2]

        # 縦と横の半分の地点
        trim_h = int(h * 0.5)
        trim_w = int(w * 0.5)

        img1 = new_img[0:trim_h, 0:trim_w]
        self.img1 = img1
        img2 = new_img[0:trim_h, trim_w:w]
        img3 = new_img[trim_h:h, 0:trim_w]
        img4 = new_img[trim_h:h, trim_w:w]

         # 分割後の画像を保存
        np.save('devide_images/img1_4.npy',img1)
        np.save('devide_images/img2_4.npy',img2)
        np.save('devide_images/img3_4.npy',img3)
        np.save('devide_images/img4_4.npy',img4)

        return img1,img2,img3,img4


    # 座標の統合
    def integration(self , file00 , file01 ,file10 ,file11):

        devide_image = self.devide_image()
        img1,img2,img3,img4 = devide_image 
        h, w = img1.shape[:2]
        # ピクセルの座標を呼び出す
        # 左上
        point00 = np.load(file00)
        p00 = []
        for p in point00:
            xmin = p[0]
            xmax = p[2]
            ymin = p[1]
            ymax = p[3]
            p00.append( [ xmin, ymin,xmax,ymax ])


        # 右上
        point01 = np.load(file01)
        p01 = []
        for p in point01:
            xmin = w + p[0]
            xmax = w + p[2]
            ymin = p[1]
            ymax = p[3]
            p01.append( [ xmin, ymin,xmax,ymax ])


        point10 = np.load(file10)
        p10 = []
        for p in point10:
            xmin = p[0]
            xmax = p[2]
            ymin = h + p[1]
            ymax = h + p[3]
            p10.append( [ xmin, ymin,xmax,ymax ])


        point11 = np.load(file11)
        p11 = []
        for p in point10:
            xmin = w + p[0]
            xmax = w + p[2]
            ymin = h + p[1]
            ymax = h + p[3]
            p11.append( [ xmin, ymin,xmax,ymax ])

    
        # 統合した座標をひとまとめにする
        global_point = p00 + p01 + p10 + p11

        print('統合後の座標',global_point)
        # 統合後の座標を保存
        np.save('global_zahyou/image_point4.npy',global_point)

        point00 = point00.tolist()
        point01 = point01.tolist()
        point10 = point10.tolist()
        point11 = point11.tolist()
        local_point = point00+ point01+ point10+ point11
        # 変換前の座標を保存
        np.save('local_zahyou/local_point4.npy',local_point)
        
              
        
# 5分割するクラス
class Devide_image5(object):
    
    def __init__(self,image):
        self.img = cv2.imread(image)
    
    def devide_image(self):
        ## 5分割
        #    01
        # 10 11  12
        #    21
        new_img = resize_image(self.img)
        h, w = new_img.shape[:2]

        # 縦と横の半分の地点
        half_h = int(h * 0.5)
        half_w = int(w * 0.5)
        h_25 = int(h * 0.25)
        w_25 = int(w * 0.25)
        h_75 = int(h * 0.75)
        w_75 = int(w * 0.75)
        # 上
        img1 = new_img[0:half_h, w_25:w_75+1]
        # 左
        img2 = new_img[h_25:h_75+1, 0:half_w]
        # 右
        img3 = new_img[h_25:h_75+1, half_w:w]
        # 下
        img4 = new_img[half_h:h, w_25:w_75+1]
        # 中央
        img5 = new_img[h_25:h_75+1, w_25:w_75+1]
        
        # 分割後の画像を保存
        np.save('devide_images/img1_5.npy',img1)
        np.save('devide_images/img2_5.npy',img2)
        np.save('devide_images/img3_5.npy',img3)
        np.save('devide_images/img4_5.npy',img4)
        np.save('devide_images/img5_5.npy',img5)
        
        return img1,img2,img3,img4,img5
        
        

# 座標の統合
    def integration(self , file01 , file21 ,file10 ,file12, file11):
        
        #resize = Resize(self.img)
        new_img = resize_image(self.img)
        h , w  = new_img.shape[:2]
        margin_h = h * 0.25
        margin_w = w * 0.25
        
        # 縦のうち上の方
        point01 = np.load(file01)
        p01 = []
        for p in point01:
            xmin = margin_w + p[0]
            xmax = margin_w + p[2]
            ymin = p[1]
            ymax = p[3]
            p01.append( [ xmin, ymin,xmax,ymax ])


        # 縦分割のうち下の方
        point21 = np.load(file21)
        # 上にある画像のサイズを取得
        dedvide_image = self.devide_image()
        img1,img2,img3,img4,img5 = dedvide_image
        h , w = img1.shape[:2]
        p21 = []
        for p in point01:
            xmin = margin_w + p[0]
            xmax = margin_w + p[2]
            ymin = h + p[1]
            ymax = h + p[3]
            p21.append( [ xmin, ymin,xmax,ymax ])


        #横分割のうち左のほう
        point10 = np.load(file10)
        p10 = []
        for p in point10:
            xmin = p[0]
            xmax = p[2]
            ymin = margin_h + p[1]
            ymax = margin_h + p[3]
            p10.append( [ xmin, ymin,xmax,ymax ])


        #横分割のうち右のほう
        point12 = np.load(file12)
        #左にある画像のサイズを取得
        h , w = img2.shape[:2]
        p12 = []
        for p in point12:
            xmin = w + p[0]
            xmax = w + p[2]
            ymin = margin_h + p[1]
            ymax = margin_h + p[3]
            p12.append( [ xmin, ymin,xmax,ymax ])


        # 中央エリア
        point11 = np.load(file11)
        #左にある画像のサイズを取得
        p11 = []
        for p in point21:
            xmin = margin_w + p[0]
            xmax = margin_w + p[2]
            ymin = margin_h + p[1]
            ymax = margin_h + p[3]
            p11.append( [ xmin, ymin,xmax,ymax ])
        
        
        # 統合した座標をひとまとめにする
        global_point = p01 + p10 + p12 + p21 + p11
        
        print('統合後の座標',global_point)
        # 統合後の座標を保存
        np.save('global_zahyou/image_point5.npy',global_point)

        point01 = point01.tolist()
        point21 = point21.tolist()
        point10 = point10.tolist()
        point12 = point12.tolist()
        point11 = point11.tolist()
        local_point = point01+ point21+ point10+ point12+ point11
        np.save('local_zahyou/local_point5.npy',local_point)
        
        
        
        
        
        
 # 分割したい画像を入れる
dev = Devide_image4(jumin)    
# ４分割にする
# 1 2
# 3 4
#それぞれ左から１、2、3、4の順にSSDで読み取った座標パスを入れる
dev.integration(zahyou1 , zahyou2 ,zahyou3 ,zahyou4)



# 分割したい画像を入れる
dev = Devide_image5(jumin)    
## 5分割
#    01
# 10 11  12
#    21
#それぞれ左から0１、21、10、12、11の順にSSDで読み取った座標パスを入れる
dev.integration( zahyou01 , zahyou21 ,zahyou10 ,zahyou12, zahyou11)