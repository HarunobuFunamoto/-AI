# coding:utf-8

# 学習用テスト用データ
# jpg画像にして保存

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import cv2,random,os
import numpy as np
from sklearn import model_selection
#import matplotlib.pyplot as plt

alfabet_li = 'zxcvbnmasdfghjklqwertyuiopＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'.replace('',' ').split()
#alfabet_li = ['Ｄ']


# 漢字ごとにフォルダに分けて、漢字ごとのフォルダ内に画像を何枚作るか
# nb *3 = 枚作る
# 一つのフォントで２種類のノイズ画像を作る
#nb = 1
nb = 1000



# 作成する画像のサイズ指定（正方形）
image_size = 100
# 回転する角度の範囲を指定
ang_min= 0
ang_max = 0
# 以下の範囲内でランダムに画像を拡大、縮小。
#画像の拡大率
size_min = 130
size_max = 140


# 平行移動
tate =0
yoko =0 

# フォント５種類
#font_list = ['JISZ8903-Medium.otf' , 'TTEdit2by3Gothic.ttf' , 'TTEditHalfGothic.ttf' , 'irohamaru-Medium.ttf']
#font_list = ['irohamaru-Medium.ttf']
font_list = [ 'W3.ttc' ]


def font_image(fontname ,mj , num_image ):
    moji_list = []
    label = []
    fontsize = int(image_size*0.7)
    for moji in mj:
        
        for num in range(num_image):
            
            image=Image.new('L', (image_size,image_size), 'white')
            image = image.convert("RGB") 
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(fontname, fontsize,encoding='utf-8')

            moji = str(moji)

            # 真ん中に配置
            img_size = np.array(image.size)
            txt_size = np.array(font.getsize(moji))
            pos = (img_size - txt_size) / 2

            draw.text(pos, moji, font=font, fill='#000')

            img = np.array(image)

            # がウシアンノイズ
            mean = 100
            std =40
            img = img +np.random.normal(mean,std,img.shape)
            img = np.clip(img ,0,200)
        

            #画像のサイズ
            scale = random.randint(size_min,size_max)
            scale = scale / 100

            #　平行移動
            dy =random.randint(0,tate)
            dx =random.randint(0,yoko)

            # 回転角度
            ang = random.randint(ang_min , ang_max)


           #  回転、拡大縮小、平行移動を実施
            M = cv2.getRotationMatrix2D( (image_size/2 , image_size/2), ang,scale)
            dst = cv2.warpAffine(img,M,(image_size,image_size))
            M1 = np.float64([[1,0,dx],[0,1,dy]])
            dst2 = cv2.warpAffine(dst,M1,(image_size,image_size))



            # ブラー
            dst2 = cv2.blur(dst2 ,(5,5))
            
            dst2 = cv2.blur(dst2 ,(8,8))

            
            moji_list.append(dst2)
            #label.append(idx)
                  
    return moji_list


def font_image2(fontname , mj , num_image ):
    moji_list = []
    label = []
    fontsize = int(image_size*0.7)
    for moji in mj:
        

        for num in range(num_image):
            image=Image.new('L', (image_size,image_size), 'white')
            image = image.convert("RGB") 
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(fontname, fontsize,encoding='utf-8')

            moji = str(moji)

            # 真ん中に配置
            img_size = np.array(image.size)
            txt_size = np.array(font.getsize(moji))
            pos = (img_size - txt_size) / 2

            draw.text(pos, moji, font=font, fill='#000')

            img = np.array(image)

        

            #画像のサイズ
            scale = random.randint(size_min,size_max)
            scale = scale / 100

            #　平行移動
            dy =random.randint(0,tate)
            dx =random.randint(0,yoko)

            # 回転角度
            ang = random.randint(ang_min , ang_max)


           #  回転、拡大縮小、平行移動を実施
            M = cv2.getRotationMatrix2D( (image_size/2 , image_size/2), ang,scale)
            dst = cv2.warpAffine(img,M,(image_size,image_size))
            M1 = np.float64([[1,0,dx],[0,1,dy]])
            dst2 = cv2.warpAffine(dst,M1,(image_size,image_size))



            # ガウシアンブラー
            dst2 = cv2.blur(dst2 ,(5,5))
            dst2 = cv2.blur(dst2 ,(5,5))
            dst2 = cv2.blur(dst2 ,(5,5))
            dst2 = cv2.blur(dst2 ,(8,8))

            
            moji_list.append(dst2)
            #label.append(idx)
                  
    return moji_list


def font_image3(fontname , kanji , num_image ):
    moji_list = []
    label = []
    fontsize = int(image_size*0.7)
    for moji in kanji:

        for num in range(num_image):

            image=Image.new('L', (image_size,image_size), 'white')
            image = image.convert("RGB") 
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(fontname, fontsize,encoding='utf-8')

            moji = str(moji)

            # 真ん中に配置
            img_size = np.array(image.size)
            txt_size = np.array(font.getsize(moji))
            pos = (img_size - txt_size) / 2

            draw.text(pos, moji, font=font, fill='#000')

            img = np.array(image)
            
            #label.append(idx)

            #画像のサイズ
            scale = random.randint(size_min,size_max)
            scale = scale / 100

            #　平行移動
            dy =random.randint(0,tate)
            dx =random.randint(0,yoko)

            # 回転角度
            ang = random.randint(ang_min , ang_max)


           #  回転、拡大縮小、平行移動を実施
            M = cv2.getRotationMatrix2D( (image_size/2 , image_size/2), ang,scale)
            dst = cv2.warpAffine(img,M,(image_size,image_size))
            M1 = np.float64([[1,0,dx],[0,1,dy]])
            dst2 = cv2.warpAffine(dst,M1,(image_size,image_size))

            moji_list.append(dst2)
            #label.append(idx)
                  
    return moji_list




    
# 漢字ごとにフォルダとothersのフォルダを作りそれぞれに画像を作る
for font in font_list:
    for alfabet in alfabet_li: 
        moji_list= font_image(font, alfabet, nb)
        moji_list = np.array(moji_list)
        

        if not os.path.exists(alfabet) :
            os.makedirs(alfabet)
        for i , m in enumerate(moji_list):
            cv2.imwrite(alfabet+'/'+alfabet+'_'+str(i)+'_'+str(font)+'_noise.jpg' , m)
        

        moji_list2= font_image2(font, alfabet, nb)
        moji_list2 = np.array(moji_list2)


        if not os.path.exists(alfabet) :
            os.makedirs(alfabet)
        for i , m in enumerate(moji_list2):
            cv2.imwrite(alfabet+'/'+alfabet+'_'+str(i)+'_'+str(font)+'_noise2.jpg' , m)
        

        moji_list3= font_image3(font, alfabet, nb)
        moji_list3 = np.array(moji_list3)


        if not os.path.exists(alfabet) :
            os.makedirs(alfabet)
        for i , m in enumerate(moji_list3):
            cv2.imwrite(alfabet+'/'+alfabet+'_'+str(i)+'_'+str(font)+'.jpg' , m)
        

        print(alfabet + '作成完了')


mojinum = len(moji_list)+len(moji_list2)+len(moji_list3)
print(alfabet+ '画像枚数',mojinum * len(font_list))  
   

