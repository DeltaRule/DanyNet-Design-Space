# DanyNet-Design-Space

Based on this Pape:r https://arxiv.org/abs/2003.13678 I designed my own Design Space
The first test is about Normalization Technique and Art of Convolutions. You can find it under Version 0<br/>

## Version 0

The 3 Normalization Taktiks are:<br/>
Batch Normalization<br/>
<img src= "./outputImages/batchnomalization_Version0.png">
Weight Normalization<br/>
<img src= "./outputImages/weightnomalization_Version0.png">
Instance Normalization<br/>  
<img src= "./outputImages/instancenomalization_Version0.png">
There you can see that Batch Normalization is still the best Technique and that weight normalization seems to be better for this Example then Instance Normalization

The three arts of ConvBlocks are:<br/>
ResnetBlock with 2 Convolutions without Groups (1)<br/>
<img src= "./outputImages/noGroups_Version0.png">
ResnetBlock with 2 Convolutions first being a depthwise Convolution and second no Groups(2)<br/>
<img src= "./outputImages/Dec First_Version0.png">
ResnetBlock with 2 Convolutions first no Groups second being the depthwise Convolution.(3)<br/>
<img src= "./outputImages/Dec Second_Version0.png">
It Seems that around a 30% (2) and 70% (1) is the best option. But this doesn't seem to matter that much. The only thing that is slighly worese is the usage of (3).<br/>
The Design space rating via EDF looks like this:<br/>
<img src= "./outputImages/DesignSpaceEDF_Version0.png">
