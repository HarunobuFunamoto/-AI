#モデルと重みと免許証を切り取った画像をテスト用にして文字出力

from keras.models import Sequential,Model
from keras.layers import Convolution2D, MaxPooling2D,Input
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.utils import np_utils
import cv2,sys
from keras.models import model_from_json
from create_allmoji import cre_moji




# 使用する重みのファイル名
#filename = sys.argv[1]
# 使用するモデルのファイル名
modelname = 'cnn_model_weight/model.json'
# 使用する重みのファイル名
weight = 'cnn_model_weight/CNN_.00-0.00.hdf5'
# imagesに予測したいnpyファイルを与える=切り取った免許証の画像　
images = 'ssd_trim_img/trim_menkyo.npy'

moji_list = cre_moji()

label_moji= {}
for idx, moji in enumerate(moji_list):
     label_moji[idx] = moji




img = np.load(images)
np_img = np.array(img)


    
# モデルと重みをロードする
model = model_from_json(open(modelname).read())
model.compile(loss='binary_crossentropy',
                   optimizer='Adam')

model.load_weights(weight)

# 予測
pred= model.predict(np_img)


predicted = np.argmax(pred, axis=1)



predicted = np.array(predicted)

#print('\n正解',' 　', '予想', sep='\t')
for pre in predicted:
    print(label_moji[pre])
