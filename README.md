MULTI MODAL DEEP LEARNING FOR SEMANTIC SEGMENTATION USING KERAS

Architecture Reference (first two models in this link): http://deepscene.cs.uni-freiburg.de/index.html

Dataset Reference (Freiburg forest multimodal/spectral annotated): http://deepscene.cs.uni-freiburg.de/index.html#datasets

My project involves using Near Infrared and RGB modalities.
The corresponding image files in both the RGB and NIR directories must have the same name and a text file containing all the image file names WITHOUT the extension (ex: imagename.jpg -> wrong || imagename -> correct).
The path to the text file will be mentioned in the comments of the code.


This following files in the repository ::

1.Deepscene/nir_rgb_segmentation_arc_1.py :: ("CHANNEL-STACKING MODEL") 
2.Deepscene/nir_rgb_segmentation_arc_2.py :: ("LATE-FUSION MODEL")
3.Deepscene/nir_rgb_segmentation_arc_3.py :: ("Convoluted Mixture of Deep Experts (CMoDE) Model")

are the exact replicas of the architectures described in Deepscene website.

The files:
1.

# NOTE:
Data augmentation is not done in this code, but it is implemented in the NEXT FILE('nir_rgb_segmentation_arc_2.py').
The augmentation part is same for both the codes, so you can simply copy paste that part to this code if needed.


2.nir_rgb_segmentation_arc_2.py :: which corresponds to "LATE-FUSED CONVOLUTION MODEL" in the mentioned reference above

3.The third architecute has been implemented but not yet tested | the model compiles perfectly | run time errors are possible. 
