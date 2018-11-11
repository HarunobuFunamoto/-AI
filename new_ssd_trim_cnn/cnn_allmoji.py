from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.utils import np_utils
import cv2,os
from PIL import Image
import glob
import keras.backend as K
from sklearn import cross_validation , metrics
from sklearn.metrics import confusion_matrix,f1_score ,recall_score ,precision_score,classification_report
from keras.callbacks import ModelCheckpoint
from create_allmoji import cre_moji




moji_list = cre_moji()
nb_moji = len(moji_list)


batch_size = 1

epoch = 1

baseSaveDir = "cnn_model_weight/"


label_moji= {}
for idx, moji in enumerate(moji_list):
     label_moji[idx] = moji
        

            

# 学習データとテストデータを分ける --- (※3)
data_train, data_test, label_train, label_test = np.load('all_moji/all_moji.npy')

data_train2 = np.array(data_train)
data_test2  = np.array(data_test)
label_train2 = np.array(label_train)
label_test2 = np.array(label_test)


#  正規化
data_train = data_train2.astype("float") / 256
data_test  = data_test2.astype("float") / 256
label_train = np_utils.to_categorical(label_train2, nb_moji)
label_test = np_utils.to_categorical(label_test2, nb_moji)



def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,border_mode='same',input_shape=in_shape)) # ----(*2a)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.7))

    model.add(Flatten()) # --- (※3)
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(nb_moji)) # ---- (*3a)
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
    optimizer='Adam',
    metrics=['accuracy'])

    return model
              


model = build_model(data_train.shape[1:])

chkpt = os.path.join(baseSaveDir, 'CNN_.{epoch:02d}-{loss:.2f}.hdf5')
cp_cb = ModelCheckpoint(filepath = chkpt, monitor='loss', verbose=1, save_best_only=True, mode='auto')


model.fit(data_train, label_train, batch_size = batch_size, nb_epoch = epoch,callbacks=[cp_cb])
score = model.evaluate(data_test, label_test)
pred = model.predict(data_test)
predicted = np.argmax(pred, axis=1)


model_json_str = model.to_json()
open(baseSaveDir + 'model.json', 'w').write(model_json_str)




print(len(label_test))


print('loss = ' , score[0])
print('accuracy =', score[1])
print('\nreport\n' , classification_report(np.argmax(label_test, axis=1), predicted))
print('\nconfusion matrix\n',confusion_matrix(np.argmax(label_test, axis=1), predicted))


predicted = np.array(predicted)
predicted_moji=[]
answer_moji=[]
for p in predicted: 
    pre_moji = label_moji[p]
    predicted_moji.append(pre_moji)
    
for lbl in label_test2:    
    lbl_moji = label_moji[lbl]
    answer_moji.append(lbl_moji)
    
print('\n正解',' 　', '予想', sep='\t')

for n in range(len(label_test) ):
    print(answer_moji[n],'　　',predicted_moji[n], sep='\t')