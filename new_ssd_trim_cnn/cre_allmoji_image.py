from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import cv2,random
import numpy as np
from sklearn import cross_validation
from kanji_create import cre_kanji
from create_allmoji import cre_moji



#  漢字のリストが格納されている
text = cre_moji()
# 作成する画像のサイズ指定（正方形）
image_size = 200
# 回転する角度の範囲を指定
ang_min= -100
ang_max = 100
# 以下の範囲内でランダムに画像を拡大、縮小。
#画像の縮小率
size_min = 60
# 画像の拡大率はこのままで
size_max = 100
# 一文字につき、何枚回転、平行移動、縮小した画像を作成するか指定。回転も移動も縮小もしない画像は1文字につき1枚は作成していて、
#それ以外に何枚画像を変形させるか指定する
num_image = 1
tate =50
yoko =50 


moji_list = []
label = []

fontsize = int(image_size*0.7)
for idx, moji in enumerate(text):
    image=Image.new('L', (image_size,image_size), 'white')
    image = image.convert("RGB") 
    draw = ImageDraw.Draw(image)
    fontname = 'TTEdit2by3Gothic.ttf'
    font = ImageFont.truetype(fontname, fontsize,encoding='utf-8')
    
    moji = str(moji)

    # 真ん中に配置
    img_size = np.array(image.size)
    txt_size = np.array(font.getsize(moji))
    pos = (img_size - txt_size) / 2

    draw.text(pos, moji, font=font, fill='#000')
    
    img = np.array(image)
    moji_list.append(img)
    label.append(idx)
                      
    for num in range(num_image):
        
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
        label.append(idx)


data_train, data_test, label_train, label_test = cross_validation.train_test_split(moji_list, label)
xy = (data_train, data_test, label_train, label_test)
np.save('all_moji/all_moji.npy', xy)
print('完了')