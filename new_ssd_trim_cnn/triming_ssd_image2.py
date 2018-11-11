import cv2
import numpy as np
import matplotlib.pyplot as plt


image_size = 200


images = np.load('trim/trimming_images.npy')
image_point = np.load('trim/image_point.npy')


trimming_img = []

for img in images:
    #  画像ごとにまとめた座標
    for img_point in image_point:
        # zahyou = [x1,y1,x2,y2]
        # x1,y1は左上の座標、x2,y2は右上の座標
        for zahyou in img_point:
        
            trim_img = img[zahyou[1]:zahyou[3], zahyou[0]:zahyou[2]]
            
            # なぜか切り取った後にxかyの値が０ものが出てくるので、それは除外
            if trim_img.shape[0] == 0 or trim_img.shape[1] == 0:
                del trim_img
                continue
            
            t_img = cv2.resize(trim_img,(image_size, image_size))
            
            #plt.imshow(cv2.cvtColor(trim_img , cv2.COLOR_BGR2RGB))
            #plt.show()  
            
            trimming_img.append(t_img)
            
# これを文字認識CNNで予測用としてロード            
np.save("ssd_trim_img/trim_menkyo.npy", trimming_img)
print('OK')

