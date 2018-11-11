# coding: utf-8
#  一文字だけの判定



def train_1moji(m):

    
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
    from keras.layers.normalization import BatchNormalization
    from keras import regularizers
    from kanji_list import cre_kanji
    from keras.optimizers import Adam
    #from keras.callbacks import History
    
    # すべての文字を
    #weight = 'good_weight/CNN_.45-0.06.hdf5'

    batch_size = 100
    
    epoch = 200
    #epoch = 3
    image_size = 32
    baseSaveDir = "weight/"
    baseSaveModel = 'model/'

    # このフォルダから予測したい画像を取得する
    test_file = 'test_higashi2'

    # このフォルダから学習用の画像を取得する
    image_dir = 'num_images/'

    moji = [m , 'others']

    model_name = baseSaveModel + moji[0]+ '_model.json'
    weight_name = baseSaveDir + moji[0]+'_weight.h5'
    #image_dir = 'hiragana_images/'


    data_train = []
    data_test = []
    label_train = []
    label_test = []
    index = []
    
        
    files = glob.glob(image_dir + m + "/*.jpg")
    #files = glob.glob(m + "/*.jpg")
    #print(files)
    for i , file in enumerate(files):
        filename = os.path.basename(file)
        img = Image.open(file)
        img = img.convert('RGB')
        img_array = np.array(img)
        data_train.append(img_array)
        label_train.append(1)
        search = moji[1] + "/*.jpg"
        #print(search)
        global all_files
        all_files = glob.glob(search)
        for ix , other_file in enumerate(all_files):
            other_filename = os.path.basename(other_file)
            if filename == other_filename:
                index.append(ix) 
    print(m+'のラベルは１')

    #print(filename)
    search = moji[1] + "/*.jpg"                
    all_files = glob.glob(search)
    for ind in index:
        all_files[ind] = ''
    f = filter(lambda s:s != '', all_files)
    all_file = list(f)
    #print(all_file)
    for file in all_file: 
        img = Image.open(file)
        img = img.convert('RGB')
        img_array = np.array(img)
        data_train.append(img_array)
        label_train.append(0)

    print(moji[1]+'のラベルは０')
    #print(label_train)


# テストデータ
    files = glob.glob(test_file + "/*.jpg")

    for i , file in enumerate(files):
        filename = os.path.basename(file)
        img = Image.open(file)
        img = img.convert('RGB')
        img_array = np.array(img)
        data_test.append(img_array)
    # 先頭文字だけを抜き取り
        file_moji = filename[:1]
        
        if moji[0] in file_moji:
            label_test.append(1)
        else:
            label_test.append(0)
    
    print('Label_test',label_test)


    nb_moji = len(moji)


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
        model.add(Convolution2D(32, 3, 3,border_mode='same',input_shape=in_shape,init='he_normal'))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.7))

        model.add(Convolution2D(64, 3, 3, border_mode='same',init='he_normal'))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        model.add(Dropout(0.7))

        model.add(Convolution2D(64, 3, 3,border_mode='same',init='he_normal'))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.7))

        model.add(Flatten()) # --- (※3)
        model.add(Dense(512 ,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones'))          
        model.add(Activation('relu'))
        model.add(Dropout(0.7))
        model.add(Dense(nb_moji)) # ---- (*3a)
        model.add(Activation('softmax'))

        adam = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

        return model



    model = build_model(data_train.shape[1:])

    #chkpt = os.path.join(baseSaveDir, 'CNN_.{epoch:02d}-{loss:.2f}.hdf5')
    #cp_cb = ModelCheckpoint(filepath = chkpt, monitor='loss', verbose=1, save_best_only=True, mode='auto')

    #model.load_weights(weight)

    history = model.fit(data_train, label_train, batch_size = batch_size, nb_epoch = epoch ,  validation_data=(data_test, label_test))
    score = model.evaluate(data_test, label_test)
    pred = model.predict(data_test)
    predicted = np.argmax(pred, axis=1)
    pre_max = np.amax(pred, axis=1)


    #plt.plot(history.history['acc'],color = 'orange')
    #plt.plot(history.history['val_acc'],color = 'blue')
    #plt.title('model accuracy')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    

    




    model_json_str = model.to_json()
    open(model_name, 'w').write(model_json_str)
    print(model_name+' '+'model保存完了')

    model.save_weights(weight_name)
    print(weight_name+' '+'weight保存完了')


    np.set_printoptions(suppress=True)
    print('predict_all%\n')
    print(pred)
    print('\n')
    #print('predicted',predicted)
    for i , pre_num in enumerate(predicted):
        if pre_num == 1:
            predict_percent = pred[i]
            print('predict_'+str(moji[0])+'%',predict_percent[1])
        else:
            print('others')

    #print('predict_max%\n')
    #print(pre_max)
    print('\n')

    print('loss = ' , score[0])
    print('accuracy =', score[1])
    print('\nreport\n' , classification_report(np.argmax(label_test, axis=1), predicted))
    print('\nconfusion matrix\n',confusion_matrix(np.argmax(label_test, axis=1), predicted))


    predicted = np.array(predicted)
    predicted_moji=[]
    answer_moji=[]


    for p in predicted: 
        predicted_moji.append(p)

    for lbl in label_test2:    
        answer_moji.append(lbl)

    print('\n正解',' 　', '予想')

    for n in range(len(predicted_moji) ):
        print(answer_moji[n],'　　',predicted_moji[n])



    


        
if __name__ == '__main__':
    from multiprocessing import Pool
    from contextlib import closing
    from kanji_list import cre_kanji
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    #from keras.callbacks import History

    all_moji = [str(num) for num in range(10)]
    repeat = 1

    for i in range(repeat):
        for moji in all_moji:
            print(moji+'を特定するための学習中')
            train_1moji(moji)

    #plt.savefig('epochs_diagnostics_number.png')

    
