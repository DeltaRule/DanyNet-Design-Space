# DanyNet-Design-Space

Based on this Pape:r https://arxiv.org/abs/2003.13678 I designed my own Design Space
The first test is about Normalization Technique and Art of Convolutions. You can find it under Version 0<br/>
The 3 Normalization Taktiks are:<br/>
Batch Normalization<br/>
Weight Normalization<br/>
Instance Normalization<br/>  
There you can see that Batch Normalization is still the best Technique and that weight Decay seems to be better for this Example then Instance Normalization
The three arts of ConvBlocks are:<br/>
ResnetBlock with 2 Convolutions without Groups (1)<br/>
ResnetBlock with 2 Convolutions first with Groups = filter and second no Groups(2)<br/>
ResnetBlock with 2 Convolutions first no Groups second with Groups = amount of filter.(3)<br/>

It Seems that around a 30% (2) and 70% (1) is the best option. But this doesn't seem to matter that much. The only thing that is slighly worese is the usage of (3).
