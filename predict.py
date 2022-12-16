import os,cv2,keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications.vgg16 import VGG16
#with strategy.scope():
vggmodel1 = VGG16(weights='imagenet', include_top=True)
vggmodel2 = VGG16(weights='imagenet', include_top=True)
vggmodel3 = VGG16(weights='imagenet', include_top=True)
vggmodel4 = VGG16(weights='imagenet', include_top=True)
vggmodel5 = VGG16(weights='imagenet', include_top=True)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

X1= vggmodel1.layers[-2].output
X2= vggmodel2.layers[-2].output
X3= vggmodel3.layers[-2].output
X4= vggmodel4.layers[-2].output
X5= vggmodel5.layers[-2].output

predictions_g= Dense(2, activation="softmax")(X1)
predictions_m = Dense(2, activation="softmax")(X2)
predictions_p = Dense(2, activation="softmax")(X3)
predictions_t = Dense(2, activation="softmax")(X4)
predictions_z = Dense(2, activation="softmax")(X5)

model_gas = Model(vggmodel1.input, predictions_g)
model_mineral = Model(vggmodel2.input, predictions_m)
model_protos = Model(vggmodel3.input, predictions_p)
model_terran = Model(vggmodel4.input, predictions_t)
model_zerg = Model(vggmodel5.input, predictions_z)

opt = Adam(lr=0.0001)
model_gas.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model_mineral.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model_protos.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model_terran.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])
model_zerg.compile(loss = keras.losses.categorical_crossentropy, optimizer = opt, metrics=["accuracy"])

model_gas.summary()
model_mineral.summary()
model_protos.summary()
model_terran.summary()
model_zerg.summary()

model_gas.load_weights('/home/lordcocoro2004/rcnn/rcnn/ieeercnn_vgg16_1gas.h5')
model_mineral.load_weights('/home/lordcocoro2004/rcnn/rcnn/ieeercnn_vgg16_1mineral.h5')
model_protos.load_weights('/home/lordcocoro2004/rcnn/rcnn/ieeercnn_vgg16_1protosbase.h5')
model_terran.load_weights('/home/lordcocoro2004/rcnn/rcnn/ieeercnn_vgg16_1terranbase.h5')
model_zerg.load_weights('/home/lordcocoro2004/rcnn/rcnn/ieeercnn_vgg16_1zergbase.h5')

z=0

for e,i in enumerate(sorted(os.listdir("Dataset/video"))):
    if(not os. path. isfile('Dataset/video_out_2/rec_'+i) and i.startswith("Comp")):
        st = time.time()
        print(i)
        z += 1
        img = cv2.imread(os.path.join("Dataset/video",i))
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = img.copy()
        
        count_g = 0
        count_m = 0
        count_p = 0
        count_t = 0
        count_z = 0
        
        for e,result in enumerate(ssresults):
            if e < 2000:
                x,y,w,h = result
                timage = imout[y:y+h,x:x+w]
                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                
                out= model_gas.predict(img)                
                if out[0][0] > 0.70:
                    print(out[0][0],count_g,'gas')
                    cv2.rectangle(imout, (x, y), (x+w, y+h), (9, 247, 199), 1, cv2.LINE_AA)
                    count_g=count_g+1
                    
                out1= model_mineral.predict(img)
                if out1[0][0] > 0.70:
                    print(out1[0][0],count_g,'mineral')
                    cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 230, 255), 1, cv2.LINE_AA)
                    count_m=count_m+1
                    
                out2= model_protos.predict(img)
                if out2[0][0] > 0.70:
                    print(out2[0][0],count_g,'protos')
                    cv2.rectangle(imout, (x, y), (x+w, y+h), (255, 0, 0), 1, cv2.LINE_AA)
                    count_p=count_p+1
                    
                out3= model_terran.predict(img)
                if out3[0][0] > 0.70:
                    print(out3[0][0],count_g,'terran')
                    cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 14, 255), 1, cv2.LINE_AA)
                    count_t=count_t+1
                    
                out4= model_zerg.predict(img)
                if out4[0][0] > 0.70:
                    print(out[0][0],count_g,'zerg')
                    cv2.rectangle(imout, (x, y), (x+w, y+h), (28, 255, 0), 1, cv2.LINE_AA)
                    count_z=count_z+1
                if count_g==7 or count_m==7 or count_p==7 or count_t==7 or count_z==7:
                    break
                if (time.time() - st) > 120 :
                    break
        cv2.imwrite('Dataset/video_out_2/rec_'+i, imout)
        print('Elapsed time = {}'.format(time.time() - st))
        #break