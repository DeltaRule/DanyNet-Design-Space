import torch 
import torch.nn as nn 

class ModelParse(nn.Module):
    def __init__(self, lstToParse:list, firstFilterUp:int, nFeatures:int = 100):
        """Initilize Class that can parse through our custom list

        Args:
            lstToParse (list): The List that creates the Model with the format
            [["normalization":str, filters:int,kernelSize:int, "activation":str, typeConvolutionBlock:int, sne:bool]]
            firstFilterUp (int): the first Convolution to upgrade the filternumber
            nFeatures (int): the number of output Features
        """
        super(ModelParse,self).__init__()
        model = []
        #the start input size
        firstInputSize = 3
        model.append(nn.Conv2d(firstInputSize, firstFilterUp, kernel_size = 3, padding = 1))
        firstInputSize = firstFilterUp
        for norm,filt,kernel,act,typeConv,sne in lstToParse:
            if(typeConv == "stride"):
                model.append(strideBlock(norm,firstInputSize, filt,kernel,act))
                firstInputSize = filt
                
            else:
                
                model.append(CustomBlock(norm,filt,kernel,act,typeConv,sne))
                firstInputSize = filt
        model.append(nn.AdaptiveAvgPool2d(1))
        model.append(nn.Flatten())
        model.append(nn.Linear(firstInputSize,nFeatures))
        self.model = nn.Sequential(*model)
    def forward(self,x):
        return self.model(x)

class strideBlock(nn.Module):
    def __init__(self, norm: str, filtIn:int, filtOut:int,kernel:int, activation:str ):
        """Initilize strideBlock

        Args:
            norm (str): the normalizations used for the Block can be "weight", "instance", "batch"
            filtIn (int): in Filter
            filtOut (int): out Filter
            kernel (int): kernelsize can be 3 or 5
            activation (str):swish or Relu
        """
        super(strideBlock,self).__init__()
        model = []
        
        if(norm == "weight"):
            if(activation == "swish"):
                model.append(nn.Hardswish(inplace=True))
            else:
                model.append(nn.ReLU(inplace=True))
            model.append(nn.utils.weight_norm(nn.Conv2d(filtIn,filtOut, kernel_size=kernel, padding= 1 if kernel == 3 else 2, stride= 2 )))
        elif(norm == "instance"):
            model.append(nn.InstanceNorm2d(filtIn))
            if(activation == "swish"):
                model.append(nn.Hardswish(inplace=True))
            else:
                model.append(nn.ReLU(inplace=True))
            model.append(nn.Conv2d(filtIn,filtOut, kernel_size=kernel, padding= 1 if kernel == 3 else 2, stride= 2 ))
        else:
            model.append(nn.InstanceNorm2d(filtIn))
            if(activation == "swish"):
                model.append(nn.Hardswish(inplace=True))
            else:
                model.append(nn.ReLU(inplace=True))
            model.append(nn.Conv2d(filtIn,filtOut, kernel_size=kernel, padding= 1 if kernel == 3 else 2, stride= 2 ))
        self.Conv = nn.Conv2d(filtIn,filtOut, kernel_size=1, stride = 2)
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x) + self.Conv(x)

        
class CustomBlock(nn.Module):
    def __init__(self,norm:str,filt:int,kernel:int ,act:str,typeConv:int,sne:bool):
        """Initilize Custom Block Class

        Args:
            norm (str): the normalizations used for the Block can be "weight", "instance", "batch"
            filt (int): the number of filters
            kernel (int): the kernel Size
            act (str): the type of Activation can be "swish" for Hardswish and "relu" for ReLu
            typeConv (int): what type of Convolution will it be it can be 0,1,2 where 0 = no groups, 1 = Groups first, 2 = Groups Second
            sne (bool): if you want to use Squeze and Excition Block
        """
        super(CustomBlock,self).__init__()
        model = []
        if(typeConv == 0):
            model.append(ConvBlock(norm,act, filt,kernel,1 ))
            model.append(ConvBlock(norm,act, filt,kernel,1 ))
        elif(typeConv == 1):
            model.append(ConvBlock(norm,act, filt,kernel, filt))
            model.append(ConvBlock(norm,act, filt,kernel,1 ))
        else:
            model.append(ConvBlock(norm,act, filt,kernel,1 ))
            model.append(ConvBlock(norm,act, filt,kernel,filt ))
        if(sne):
            model.append(SELayer(filt))

        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x) + x

def ConvBlock(norm:str, act:str,filter:int, kernel:int,groups:int):
    """Create a Convolution with activation and normalization

    Args:
        norm (str): what normalization 
        act (str): what activation
        filter (int): how many Filter
        kernel (int): Kernel Size
        groups (int): Group Size

    Returns:
        torchModel: Convolution with activation and normalization
    """
    if(norm == "weight"):
        return nn.Sequential(nn.Hardswish(inplace=True) if act == "swish" else nn.ReLU(inplace=True),
        nn.utils.weight_norm(nn.Conv2d(filter,filter, kernel_size = kernel, padding= 1 if kernel == 3 else 2,groups= groups)))
    elif(norm == "instance"):
        return nn.Sequential(nn.InstanceNorm2d(filter),
            nn.Hardswish(inplace=True) if act == "swish" else nn.ReLU(inplace=True),
        nn.Conv2d(filter,filter, kernel_size = kernel, padding= 1 if kernel == 3 else 2,groups= groups))
    else:
        return nn.Sequential(nn.BatchNorm2d(filter),
            nn.Hardswish(inplace=True) if act == "swish" else nn.ReLU(inplace=True),
        nn.Conv2d(filter,filter, kernel_size = kernel, padding= 1 if kernel == 3 else 2,groups= groups))
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    # This class was taken from https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y