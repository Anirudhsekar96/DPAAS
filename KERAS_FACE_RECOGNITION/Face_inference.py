from glob import glob
from keras import backend as k 
from PIL import Image
import numpy as np
import os
import importlib
from keras.models import load_model
import numpy


k.set_image_data_format('channels_first')


#path_test = './data_for_validation/srk_val/*'
#path_test = '../tensorflow-face-detection/media/*.png'
path_test = '/media/anirudh/Data/Code/Dpaas/tensorflow-face-detection/Full_Picture /Full_pic_1/*.png'
images_test_path = glob(path_test)
#print(images_test_path)

def preproc(fp):
    im = Image.open(fp).convert('RGB')
    im = im.resize((224,224))
    im = np.array(im).astype(np.float32)
    im = im.transpose((2,0,1))
    #im = np.expand_dims(im, axis=0)
    return im

def test_data():
    images = []
    for img in images_test_path:
        images.append(preproc(img))
        #print('loaded')
    images = np.asarray(images)
    return images

if __name__ == "__main__":

    
    #model = load_model('vgg16_face_scraped.h5')
    model = load_model('vgg16_face_scraped_personal.h5')
    print('Model Loaded')

    images = test_data()
    #print(len(images))
    out = model.predict(images,verbose=2)
    #print(out)
    '''
    for i in out:
        if np.argmax(i)==0:
            print("Aamir Khan")
        elif np.argmax(i)==1:
            print("Samantha")
        else:
            print("Shahrukh Khan")
    '''
    abc = set()
    for i in out:
        if np.argmax(i)==0:
            abc.add('14BCE1001')
        elif np.argmax(i)==1:
            abc.add('14BCE1002')
        elif np.argmax(i)==2:
            abc.add('14BCE1003')
        elif np.argmax(i)==3:
            abc.add('14BCE1004')
        elif np.argmax(i)==4:
            abc.add('14BCE1005')
        elif np.argmax(i)==5:
            abc.add('14BCE1006')
        elif np.argmax(i)==6:
            abc.add('14BCE1007')
        elif np.argmax(i)==7:
            abc.add('14BCE1008')
        elif np.argmax(i)==8:
            abc.add('14BCE1009')
        elif np.argmax(i)==9:
            abc.add('14BCE1010')
        elif np.argmax(i)==10:
            abc.add('14BCE1011')
        elif np.argmax(i)==11:
            abc.add('14BCE1012')
        elif np.argmax(i)==12:
            abc.add('14BCE1013')
        elif np.argmax(i)==13:
            abc.add('14BCE1014')
        elif np.argmax(i)==14:
            abc.add('14BCE1015')
        elif np.argmax(i)==15:
            abc.add('14BCE1016')

        #abc.add(np.argmax(i))
    
    print(abc)
