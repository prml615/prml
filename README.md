# Multimodal fully convolutional network for SEMANTIC SEGMENTATION using Keras.
Keras implementation of Fully convolutional Network (FCN-32s)trained to predict semantically segmented images of forest like images with rgb & nir_color input images.

The Following list describes the files :

1. late_fusion_improveed.py            (late_fusion FCN TRAINING FILE, Augmentation= Yes, Dropout= Yes)
2. late_fusion_improved_predict.py     (predict with improved architecture)
3. late_fusion_improved_saved_model.hdf5 (Architecture & weights of improved model)
4. late_fusion_old.py                  (late_fusion  FCN TRAINING FILE, Augmentation= No, Dropout= No)
5. late_fusion_old_predict.py()        (predict with old architecture)
6. late_fusion_improved_saved_model.hdf5 (Architecture & weights of old model)



### Architecture:
![Alt text](/Misc/Arc.png)
Architecture Reference (first two models in this link): http://deepscene.cs.uni-freiburg.de/index.html

### Dataset:
![Alt text](/Ds.png)
Dataset Reference (Freiburg forest multimodal/spectral annotated): http://deepscene.cs.uni-freiburg.de/index.html#datasets

### Training:
Loss : Categorical Cross Entropy
Optimizer : Stochastic gradient descent with lr = 0.008, momentum = 0.9, decay=1e-6

###### Note:
The corresponding image files in both the RGB and NIR directories must have the same name and a text file containing all the image file names WITHOUT the extension (ex: imagename.jpg -> wrong || imagename -> correct).
The path to the text file will be mentioned in the comments of the code.



# NOTE:


This following files in the repository ::

1.Deepscene/nir_rgb_segmentation_arc_1.py :: ("CHANNEL-STACKING MODEL") 
2.Deepscene/nir_rgb_segmentation_arc_2.py :: ("LATE-FUSION MODEL")
3.Deepscene/nir_rgb_segmentation_arc_3.py :: ("Convoluted Mixture of Deep Experts (CMoDE) Model")

are the exact replicas of the architectures described in Deepscene website.

The files:
1.



Data augmentation is not done in this code, but it is implemented in the NEXT FILE('nir_rgb_segmentation_arc_2.py').
The augmentation part is same for both the codes, so you can simply copy paste that part to this code if needed.


2.nir_rgb_segmentation_arc_2.py :: which corresponds to "LATE-FUSED CONVOLUTION MODEL" in the mentioned reference above

3.The third architecute has been implemented but not yet tested | the model compiles perfectly | run time errors are possible. 
