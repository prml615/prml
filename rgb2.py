#FOR MODIFYING IMAGES AND ARRAYS
import os, cv2
import numpy as np
#KERAS IMPORTS
import keras
from keras.applications.vgg16 import VGG16
from keras.callbacks import ProgbarLogger, EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Conv2DTranspose, Conv2D, core
from keras.preprocessing.image import *

#UTILITY GLOBAL VARIABLES
input_dim = [512, 960]  
num_class = 6
C = 10
index = [2380, 1020,  969,  240, 2775,    0]

#HELPER FUNCTION OF SEGMENT_DATA_GENERATOR
# comprises of path and extension of images in a directory
class gen_args:
    data_dir = None
    data_ext = None
    def __init__(self,dirr,ext):
        self.data_dir = dirr
        self.data_ext = ext
        

#RESIZES 3D IMAGES(image)(EX: RGB) TO DESIRED SIZE(crop_size) 
def fix_size(image, crop_size):
    cropy, cropx = crop_size
    height, width = image.shape[:-1]
    
    #adjusting height of the image 
    cy = cropy - height
    if cy > 0:
        if cy % 2 == 0:
            image = np.vstack((np.zeros((cy/2,width,3)) , image , np.zeros((cy/2,width,3))))
        else:
            image = np.vstack((np.zeros((cy/2,width,3)) , image , np.zeros((cy/2 +1,width,3))))
    if cy < 0:
        if cy % 2 == 0:
            image = np.delete(image, range(-1*cy/2), axis = 0)
            image = np.delete(image, range(height + cy,height +  cy/2), axis = 0)
        else:
            image = np.delete(image, range(-1*cy/2), axis =0)
            image = np.delete(image, range(height + cy, height + cy/2 + 1), axis=0)
    
    #adjusting width of the image
    height, width = image.shape[:-1]
    cx = cropx - width
    if cx > 0:
        if cx % 2 == 0:
            image = np.hstack((np.zeros((height,cx/2,3)) , image , np.zeros((height,cx/2,3))))
        else:
            image = np.hstack((np.zeros((height,cx/2,3)) , image , np.zeros((height,cx/2 + 1,3))))
    if cx < 0:
        if cx % 2 == 0:
            image = np.delete(image, range(-1*cx/2), axis = 1)
            image = np.delete(image, range(width + cx,width +  cx/2), axis = 1)
        else:
            image = np.delete(image, range(-1*cx/2), axis =1)
            image = np.delete(image, range(width + cx, width + cx/2 + 1), axis=1)
    return image

#====================================================data==augmentation==============================================================
class aug_state:
    def __init__(self,flip_axis_index=0,rotation_range=360,height_range=0.2,width_range=0.2,shear_intensity=1,color_intensity=40,zoom_range=(1.2,1.2)):
        self.flip_axis_index=flip_axis_index
        self.rotation_range=rotation_range
        self.height_range=height_range
        self.width_range=width_range
        self.shear_intensity=shear_intensity
        self.color_intensity=color_intensity
        self.zoom_range=zoom_range


def data_augmentor(x,state,row_axis=0,col_axis=1,channel_axis=-1,
    bool_flip_axis=True,
    bool_random_rotation=True,
    bool_random_shift=True,
    bool_random_shear=True,
    bool_random_channel_shift=True,
    bool_random_zoom=True):
    if bool_flip_axis:
        flip_axis(x, state.flip_axis_index)

    if bool_random_rotation:
        random_rotation(x, state.rotation_range, row_axis, col_axis, channel_axis)

    if bool_random_shift:
        random_shift(x, state.width_range, state.height_range, row_axis, col_axis, channel_axis)

    if bool_random_shear:
        random_shear(x, state.shear_intensity, row_axis, col_axis, channel_axis)

    if bool_random_channel_shift:
        random_channel_shift(x, state.color_intensity, channel_axis)

    if bool_random_zoom:
        random_zoom(x, state.zoom_range, row_axis, col_axis, channel_axis)

    return x



#=====================================================================================================================
#DATAGENERATOR FOR MULTIMODAL SEMANTIC SEGMENTATION
def datagen(state_aug,file_path, input_args, label_args, batch_size, input_size):
    # Create MEMORY enough for one batch of input(s) + augmented input(s) & labels + augmented labels
    data = np.zeros((batch_size*2,input_size[0],input_size[1],3))
    labels = np.zeros((batch_size*2,input_size[0],input_size[1],3))
    # Read the file names
    files = open(file_path)
    names = files.readlines()
    files.close()
    # Enter the indefinite loop of generator
    while True:
        for i in range(batch_size):
            index_of_random_sample = np.random.choice(len(names))
            np.random.seed(i)
            data[i] = fix_size(cv2.imread(input_args.data_dir+names[index_of_random_sample].strip('\n')+input_args.data_ext), input_size)
            data[batch_size+i] = data_augmentor(data[i],state_aug)
            np.random.seed(i)
            labels[i] = fix_size(cv2.imread(label_args.data_dir+names[index_of_random_sample].strip('\n')+label_args.data_ext), input_size)
            labels[batch_size+i] = data_augmentor(labels[i], state_aug, bool_random_channel_shift= False)
        yield [data],[labels]

#ARGUMENTS FOR DATA_GENERATOR
Train_DEPTH_args = gen_args ('/home/captain_jack/Downloads/freiburg_forest_annotated/Otherformats/train/depth_color/','.png')
Train_NIR_args = gen_args ('/home/captain_jack/Downloads/freiburg_forest_annotated/Otherformats/train/nir_color/','.png')
Valid_DEPTH_args = gen_args ('/home/captain_jack/Downloads/freiburg_forest_annotated/Otherformats/valid/depth_color/','.png')
Valid_NIR_args = gen_args ('/home/captain_jack/Downloads/freiburg_forest_annotated/Otherformats/valid/nir_color/','.png')

state_aug = aug_state() 

train_generator = datagen(state_aug,
    file_path = '/home/captain_jack/Downloads/freiburg_forest_annotated/Otherformats/train.txt',
    input_args = Train_DEPTH_args,
    label_args = Train_NIR_args,
    batch_size= 8,
    input_size=input_dim)


valid_generator = datagen(state_aug,
    file_path = '/home//captain_jack/Downloads/freiburg_forest_annotated/Otherformats/valid.txt',
    input_args = Valid_DEPTH_args,
    label_args = Valid_NIR_args,
    batch_size= 8,
    input_size=input_dim)

#================================================MODEL_ARCHITECTURE============================================================

# RGB MODALITY BRANCH OF CNN
inputs_rgb = Input(shape=(input_dim[0],input_dim[1],3))
vgg_model_rgb = VGG16(weights='imagenet', include_top= False)
conv_model_rgb = vgg_model_rgb(inputs_rgb)
conv_model_rgb = Conv2D(1024, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
conv_model_rgb = Conv2D(1024, (3,3), strides=(1, 1), padding = 'same', activation='relu',data_format="channels_last") (conv_model_rgb)
deconv_rgb_1 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_model_rgb)
#============================================================================================================
conv_rgb_1 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_rgb_1)
dropout_rgb = core.Dropout(0.4)(conv_rgb_1)
#===============================================================================================================
deconv_rgb_2 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(dropout_rgb)
conv_rgb_2 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_rgb_2)
deconv_rgb_3 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_rgb_2)
conv_rgb_3 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_rgb_3)
deconv_rgb_4 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_rgb_3)
conv_rgb_4 = Conv2D(num_class*C, (3,3), strides=(1,1), padding = 'same', activation='relu', data_format='channels_last')(deconv_rgb_4)
deconv_rgb_5 = Conv2DTranspose(num_class*C,(4,4), strides=(2, 2), padding='same', data_format="channels_last", activation='relu',kernel_initializer='glorot_normal')(conv_rgb_4)


# DECONVOLUTION Layers
deconv_last = Conv2DTranspose(3,
                              (1,1),
                              strides=(1, 1),
                              padding='same',
                              data_format="channels_last",
                              activation='relu',
                              kernel_initializer='glorot_normal') (deconv_rgb_5)

# MODAL [INPUTS , OUTPUTS]
model = Model(inputs=[inputs_rgb], outputs=[deconv_last])
print 'compiling'
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# Save the model according to the conditions  
progbar = ProgbarLogger(count_mode='steps')
checkpoint = ModelCheckpoint("nir_rgb_segmentation_2.{epoch:02d}.hdf5", monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=1, verbose=1, mode='auto')
#haven't specified validation data directory yet


model.fit_generator(train_generator,steps_per_epoch=2000,epochs=50,callbacks=[progbar,checkpoint],validation_data = valid_generator,validation_steps = 2000)
