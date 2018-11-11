# coding: utf-8
# ロードして文字一つを予測する

from keras.models import Sequential,Model
from keras.layers import Convolution2D, MaxPooling2D,Input
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.utils import np_utils
import cv2,sys,glob
from keras.models import model_from_json
import os, re 
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.optimizers import Adam
from keras.models import load_model


# 予測したい画像を指定する
test_file = 'test_higashi2'
# ロードしたいモデルとウェイト
baseSaveDir = "weight/"
baseSaveModel = 'model/'



def predict_moji(image):

    weight_files = glob.glob(baseSaveDir  + "/*.h5")
    model_files = glob.glob(baseSaveModel + "/*.json")
    
    # ここをすべての文字にして、kanjiをmoij = []にして同じ文字同士のウェイトとモデルを用意してロード
    probability = []
    predicted_moji = []
    for idx, model_file in enumerate(model_files):
        model_filename = os.path.basename(model_file)
        file_moji = model_filename[:1]
        #print(file_moji)
        for wgt in weight_files:
            if re.search(file_moji, wgt):
                weightname = wgt
                
        for mdl in model_files:
            if re.search(file_moji, mdl):
                modelname = mdl
                

        #print('weight',weightname)
        #print('model',modelname)
        

        # モデルと重みをロードする
        model = model_from_json(open(modelname).read())
        #model = load_model('model/東my_model.h5')
        #adam = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

        model.load_weights(weightname)


        # 予測
        pred= model.predict(image)
        predicted = np.argmax(pred, axis=1)
        

        np.set_printoptions(suppress=True)
        for i , pre_num in enumerate(predicted):
            if pre_num == 1:
                predict_percent = pred[i]
                #print('predict_'+str(moji_list[idx])+'%',predict_percent[1])
                probability.append(predict_percent[1])
                # これが予測した漢字
                predicted_moji.append(file_moji[idx])
            else:
                #print('others')
                probability.append(0)
                predicted_moji.append('others')
        
    print('全ての確率',probability)
    print('全ての文字',predicted_moji)
    index = np.argmax(probability)
    conclusion = predicted_moji[index]


    print('この文字は',str(conclusion))




if __name__ == '__main__':
    import os,cv2
    import numpy as np
    from PIL import Image
    
    size = 100
    # テストデータ
    files = glob.glob(test_file + "/*.jpg")
    data_test = []
    for i , file in enumerate(files):
        filename = os.path.basename(file)
        img = Image.open(file)
        img = img.convert('RGB')
        img_array = np.array(img)
        img_array = cv2.resize(img_array , (size,size))
        img_array = img_array.astype("float") / 256
        data_test.append(img_array)

    for data in data_test:
        image = []
        image.append(data)
        image = np.array(image)
        #print(image.shape)
        predict_moji(image)



