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
test_file = 'test_higashi3'
# 予測したい画像のリサイズを指定
size = 32

# ロードしたいモデルとウェイト
baseSaveDir = "weight/"
baseSaveModel = 'model/'



def predict_moji(image,label_test):

    weight_files = glob.glob(baseSaveDir  + "/*.h5")
    model_files = glob.glob(baseSaveModel + "/*.json")

    print('正解ラベルの'+label_test+'を判定中')
    # ここをすべての文字にして、kanjiをmoij = []にして同じ文字同士のウェイトとモデルを用意してロード
    probability = []
    predicted_moji = {}
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

        print(file_moji+'予測')
        # 予測
        pred= model.predict(image)
        predicted = np.argmax(pred, axis=1)
        
        # キーに文字、値は確率
        predicted_moji[file_moji] = pred[0][1]
        #print(predicted_moji)
        
    #print('辞書型',predicted_moji)
    # 全ての確率から確率高い順に並び替え
    sort_prob = sorted(predicted_moji.items(), key=lambda x:x[1], reverse=True)
    #print(sort_prob)
    # 全ての確率からベスト５を取り出す
    best5 = sort_prob[:5]
    #print('Best5',best5)

    count = 0
    print('正解ラベルは'+label_test)
    print('予測文字','確率')
    for pre_moji , probability in best5:
        print(pre_moji ,'  ', probability)
        if label_test in pre_moji:
            count += 1
    

    return count



if __name__ == '__main__':
    import os,cv2
    import numpy as np
    from PIL import Image
    
    
    # テストデータ
    files = glob.glob(test_file + "/*.jpg")
    data_test = []
    label_moji = []
    for i , file in enumerate(files):
        filename = os.path.basename(file)
        img = Image.open(file)
        img = img.convert('RGB')
        img_array = np.array(img)
        img_array = cv2.resize(img_array , (size,size))
        img_array = img_array.astype("float") / 256
        data_test.append(img_array)
        mojiname = filename[:1]
        label_moji.append(mojiname)

    correct = 0
    for ind,data in enumerate(data_test):
        image = []
        image.append(data)
        image = np.array(image)
        lbl_test = label_moji[ind]
        pre_count = predict_moji(image,lbl_test)
        #print('count',pre_count)
        correct += pre_count
        
    accuracy = correct / len(label_moji)
    print('accuracy',str(accuracy * 100) +'%')









