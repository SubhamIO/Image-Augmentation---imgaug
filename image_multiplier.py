from PIL import Image
from imgaug import augmenters as iaa
import numpy as np
import glob,os,cv2

import warnings
warnings.filterwarnings('ignore')

path_parent = 'image/*'
path_parent = os.path.dirname(os.path.abspath(path_parent)) + '/'
dirs_parent = os.listdir(path_parent)

for item_parent in dirs_parent:
    print(item_parent)

    path = 'image/'+item_parent+'/*'
    path = os.path.dirname(os.path.abspath(path)) + '/'
    dirs = os.listdir(path)

    for item in dirs:
        if (os.path.isfile(path+item)) & ('png' in item or 'jpg' in item or 'jpeg' in item or 'PNG' in item or 'JPG' in item or 'JPEG' in item):
            im = Image.open(path+item)
            if im.mode !='RGB':
                im = im.convert('RGB')
            imResize = im.resize((250,250),Image.ANTIALIAS)
            imResize.save(path+item,'JPEG',quality = 90)
    fps = glob.glob(path+'*')
    images = np.array(
        [cv2.cvtColor(cv2.imread(fp),cv2.COLOR_BGR2RGB) for fp in fps],
        dtype = np.uint8
    )

    #======Crop,GaussianBlur,Contrast,Gaussian Noise, Lightness, Affine(rigbody transform+shear)===================
    #Link to understand Image Augmentation : https://imgaug.readthedocs.io/en/latest/source/examples_basics.html
    seq = iaa.Sequential([  #https://imgaug.readthedocs.io/en/latest/source/api_augmenters_meta.html#imgaug.augmenters.meta.Sequential
    iaa.Fliplr(1.0),#horizontal flips(https://imgaug.readthedocs.io/en/latest/source/api_augmenters_flip.html)
    iaa.Crop(percent = (0,0.1)),#random crops (https://imgaug.readthedocs.io/en/latest/source/api_augmenters_size.html#imgaug.augmenters.size.Crop)
    iaa.Sometimes(0.5, #gaussian blur with random sigma 0-0.5 in half(50% of images-Sometimes) of images (https://imgaug.readthedocs.io/en/latest/source/api_augmenters_meta.html#imgaug.augmenters.meta.Sometimes)
        iaa.GaussianBlur(sigma=(0,0.5)) #(https://imgaug.readthedocs.io/en/latest/source/api_augmenters_blur.html#imgaug.augmenters.blur.GaussianBlur)
    ),
    iaa.ContrastNormalization((0.75,1.5)),#strengthen or weaken the contrast in each image(https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html#imgaug.augmenters.arithmetic.ContrastNormalization)
    iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),#add gaussian Noise
    iaa.Multiply((0.8,1.2),per_channel=0.2),#configure Lightness(https://imgaug.readthedocs.io/en/latest/source/api_augmenters_arithmetic.html#imgaug.augmenters.arithmetic.Multiply)

    #https://imgaug.readthedocs.io/en/latest/source/api_augmenters_geometric.html#imgaug.augmenters.geometric.Affine
    iaa.Affine(    #Affine Transform : Scale/Zoom them, translate/move them, rotate them and shear them)
        scale={"x":(0.8,1.2),"y":(0.8,1.2)},
        translate_percent={"x":(-0.2,0.2),"y":(-0.2,0.2)},
        rotate=(-25,25),
        shear=(-8,8))
    ],random_order=True) #apply augmenters in random random_order

    aug_times = 8

    for times in range(aug_times):
        images_aug = seq(images=images)

        i=0
        for img in images_aug:
            cv2.imwrite(os.path.join(path,f'{times}'+'hauged_'+os.path.basename(fps[i])),cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            i+=1
