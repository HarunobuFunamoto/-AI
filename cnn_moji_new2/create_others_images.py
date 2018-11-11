# coding:utf-8

# 学習用テスト用データ
# create_others_images
# jpg画像にして保存

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import cv2,random,os
import numpy as np
from sklearn import model_selection
from kanji_list import cre_kanji




# ここで何の文字の画像を作るかいずれかを指定する（リストで指定）
# 全ての漢字
kanji_li = cre_kanji()
katakana_li = ['ァ','ア','ィ','イ','ゥ','ウ','ェ',' エ' , 'ォ','オ','カ','ガ','キ','ギ','ク','グ','ケ','ゲ','コ','ゴ','サ','ザ','シ','ジ','ス','ズ','セ','ゼ','ソ','ゾ','タ','ダ','チ','ヂ','ッ','ツ','ヅ','テ','デ','ト','ド','ナ','ニ','ヌ','ネ','ノ','ハ','バ','パ','ヒ','ビ','ピ','フ','ブ','プ','ヘ','ベ','ペ','ホ','ボ','ポ','マ','ミ','ム','メ','モ','ャ','ヤ','ュ','ユ','ョ','ヨ','ラ','リ','ル','レ','ロ','ヮ','ワ','ヰ','ヱ','ヲ','ン','ヴ','ヵ','ヶ']
hiragana_li = ['あ','い','う','え','お','か','が','き','ぎ','く','ぐ','け','げ','こ','ご','さ','ざ','し','じ','す','ず','せ','ぜ','そ','ぞ','た','だ','ち','ぢ','つ','づ','て','で','と','ど','な','に','ぬ','ね','の','は','ば','ぱ','ひ','び','ぴ','ふ','ぶ','ぷ','へ','べ','ぺ','ほ','ぼ','ぽ','ま','み','む','め','も','や','ゆ','よ','ら','り','る','れ','ろ','わ','ゐ','ゑ','を','ん','ゃ','ゅ','ょ','ぁ','ぃ','ぅ','ぇ','ぉ']
number_li = [str(num) for num in range(10)]
alfabet_li = 'zxcvbnmasdfghjklqwertyuiopＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'.replace('',' ').split()

allmoji_list = kanji_li + katakana_li + number_li + hiragana_li + alfabet_li
#allmoji_list = random.sample(allmoji_list, 2300)

# 全ての文字のうち　1000文字ランダムで選ぶ
allmoji_list = random.sample(allmoji_list, 1000)


# 全文字 * nb *3 = 枚作る
# 一つのフォントで２種類のノイズ画像を作る
nb = 1



# 作成する画像のサイズ指定（正方形）
image_size = 100
# 回転する角度の範囲を指定
ang_min= 0
ang_max = 0
# 以下の範囲内でランダムに画像を拡大、縮小。
#画像の縮小率
size_min = 100
# 画像の拡大率はこのままで
size_max = 160
# 一文字につき、何枚回転、平行移動、縮小した画像を作成するか指定。回転も移動も縮小もしない画像は1文字につき1枚は作成していて、
#それ以外に何枚画像を変形させるか指定する
# num_image 

tate =0
yoko =0 

# フォント５種類
#font_list = ['JISZ8903-Medium.otf' , 'TTEdit2by3Gothic.ttf' , 'TTEditHalfGothic.ttf' , 'irohamaru-Medium.ttf']
#font_list = ['irohamaru-Medium.ttf']
font_list = ['W3.ttc']



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
    for mj in allmoji_list: 
        moji_list= font_image(font, mj, nb)
        moji_list = np.array(moji_list)
        

        if not os.path.exists('others') :
            os.makedirs('others')
        for i , m in enumerate(moji_list):
            cv2.imwrite('others/'+mj+'_'+str(i)+'_'+str(font)+'_noise.jpg' , m)
        #print(mj+'  '+'othersへの画像枚数',len(moji_list)) 
        

        moji_list2= font_image2(font, mj, nb)
        moji_list2 = np.array(moji_list2)

            
        if not os.path.exists('others') :
            os.makedirs('others')
        for i , m in enumerate(moji_list2):
            cv2.imwrite('others/'+mj+'_'+str(i)+'_'+str(font)+'_noise2.jpg' , m)
        #print(mj+'  '+'othersへの画像枚数',len(moji_list))
        

        moji_list3 = font_image3(font, mj, nb)
        moji_list3 = np.array(moji_list3)

            
        if not os.path.exists('others') :
            os.makedirs('others')
        for i , m in enumerate(moji_list3):
            cv2.imwrite('others/'+mj+'_'+str(i)+'_'+str(font)+'.jpg' , m)
        #print(mj+'  '+'othersへの画像枚数',len(moji_list))
        

        print(mj +'  '+ '作成完了')

print('終了')


