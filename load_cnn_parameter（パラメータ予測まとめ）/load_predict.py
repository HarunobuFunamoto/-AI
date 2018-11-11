from keras.models import Sequential,Model
from keras.layers import Convolution2D, MaxPooling2D,Input
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.utils import np_utils
import cv2,sys
from keras.models import model_from_json
from gene_pre_img import gen_images


# 使用する重みのファイル名
filename = sys.argv[1]
# 使用するモデルのファイル名
modelname = 'model.json'
#  画像をランダムに回転、拡大、縮小、平行移動→それを予測する
images,label_list = gen_images(1, 280,180, 70, 130, 359,50,50)


# モデルと重みをロードする
model = model_from_json(open('cnn/'+ modelname).read())
model.compile(loss={'output1': 'mean_squared_error',
                         'output2': 'mean_squared_error',
                         'output3': 'mean_squared_error',
                         'output4': 'mean_squared_error'},
                   optimizer='Adam')

model.load_weights('cnn/'+filename)

# 予測
output1, output2,output3,output4= model.predict(images)



#label_train_size
print('\nlabel_size', 'predicted', sep='\t')
for idx, pre_label_size in enumerate(output1):
    for pre_size in pre_label_size:
            print(label_list[idx][0],' ',pre_size, sep='\t')

print('\n')


 #  label_train_row
print('\nlabel_row', 'predicted', sep='\t')

for idx, pre_label_row in enumerate(output2):
    for pre_row in pre_label_row:
            print(label_list[idx][1],' ',pre_row, sep='\t')


print('\n')


#  label_train_col
print('\nlabel_col', 'predicted', sep='\t')
for idx, pre_label_col in enumerate(output3):
    for pre_col in pre_label_col:
            print(label_list[idx][2],' ',pre_col, sep='\t')


print('\n')


#label_train_angle
print('\nlabel_angle', 'predicted', sep='\t')
for idx, pre_label_angle in enumerate(output4):
    for pre_angle in pre_label_angle:
           print(label_list[idx][3],' ',pre_angle, sep='\t')
