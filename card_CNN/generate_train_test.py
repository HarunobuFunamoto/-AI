from PIL import Image
import numpy as np
import glob
from sklearn import cross_validation


#画像を読み込んで、image_sizeでピクセルを指定して均等なサイズにリサイズ。学習用とテスト用に分けるプログラム

def gene_train_test(image_size):

    card = ['免許証','住民票','パスポート','保険証','マイナンバー']

    image_size = image_size

    data = []
    label = []
    label_card = {}
    for idx, c in enumerate(card):
        image_dir = 'images/' + c
        files = glob.glob(image_dir + "/*.jpg")
        for i , file in enumerate(files):
            img = Image.open(file)
            img = img.convert('RGB')
            img = img.resize((image_size , image_size))
            img_array = np.array(img)
            data.append(img_array)
            label.append(idx)
        label_card[idx] = c

    data_train, data_test, label_train, label_test = cross_validation.train_test_split(data, label)
    xy = (data_train, data_test, label_train, label_test)
    np.save('generate_file/card.npy', xy)
    print(label_card)
    print('学習用のデータ数：',len(data_train))
    print('テスト用のデータ数：',len(data_test))


#gene_train_test(50)
