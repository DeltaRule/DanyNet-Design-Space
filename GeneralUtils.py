from makeModel import ModelParse
import torch
import numpy as np
import time
import traceback
import torchvision
import torchvision.transforms as transforms
def createRandomModel(version:int)->dict:
    """Generate Models Based on the Version I will update this

    Args:
        version (int): wich random Factors will be variated depends on the Version

    Returns:
        dict: The Model and the model parameters
    """    
    if(version == 0):
        #the Parameters to varie are the Block types and the Normalization taktiks, this is for me personally and can't really be used for explaining the mobilenetV3
        #Standart Parameter will be RELU, Kernelsize = 3, reduktion on 8x8
        firstDepth = np.random.randint(8,21)
        whereStride = findStride(2,firstDepth)
        whereStride.append(firstDepth)
        probNorm = np.random.uniform(0,1, size = 3)
        probNorm = probNorm/sum(probNorm)
        norms = np.random.choice(["weight", "instance", "batch"], p = probNorm, size = firstDepth + 2)
        model = []
        strideTmp = 0
        startFilter = np.random.randint(16,64)
        for i in range(firstDepth):
            if(i == whereStride[strideTmp]):
                model.append([norms[i + strideTmp],startFilter*(2**(strideTmp+1)),3,"relu","stride",False])
                strideTmp += 1
            model.append([norms[i + strideTmp],startFilter*(2**(strideTmp)),3,"relu",np.random.randint(0,3),False])
        try:
            for _ in range(5):
                t = testModel(model,startFilter)
                if(t < 0.06 and t > 0.04):
                    cntNorm = np.zeros(3)
                    cntLayer = np.zeros(3)
                    for layer in model:
                        #count normalization % and layer type %
                        if(layer[0] == "weight"):
                            cntNorm[0] += 1
                        elif(layer[0] == "instance"):
                            cntNorm[1] += 1
                        else:
                            cntNorm[2] += 1 
                        try:
                            cntLayer[layer[4]] += 1
                        except:
                            pass

                    return {"stride":2, "kernel3":1, "swish":0, "sne":0, "model":model, "norms": cntNorm/sum(cntNorm), "layerType": cntLayer/sum(cntLayer),
                    "depth": len(model), "startFilter": startFilter}
                elif(t < 0.04):
                    #make model bigger when its to small
                    tmpRandom =  np.random.randint(0,2)
                    
                    
                    if(tmpRandom == 0):
                        try:
                            startFilter_new = np.random.randint(startFilter + 2,64)
                            for layerIdx in range(len(model)):
                                model[layerIdx][1] = startFilter_new * (model[layerIdx][1]//startFilter)
                            startFilter = startFilter_new
                        
                        except:
                            tmpRandom = 1
                        
                    if(tmpRandom == 1):
                        start = 0
                        for i in range(len(whereStride)-1):
                            if(start - whereStride[i] != whereStride[i] - whereStride[i+1]):
                                nor = np.random.choice(["weight", "instance", "batch"], p = probNorm)
                                model.insert(start+i, [nor,startFilter*(2**(i)),3,"relu",np.random.randint(0,3),False])
                                for j in range(i+1,len(whereStride)):
                                    whereStride[j] += 1
                                break
                            else:
                                start = whereStride[i]
                    
                else:
                    tmpRandom =  np.random.randint(0,2)
                    
                    
                    if(tmpRandom == 0):
                        try:
                            startFilter_new = np.random.randint(16,startFilter)
                            for layerIdx in range(len(model)):
                                model[layerIdx][1] = startFilter_new * (model[layerIdx][1]//startFilter)
                            startFilter = startFilter_new
                            
                        except:
                            tmpRandom = 1
                        
                    if(tmpRandom == 1 ):
                        start = whereStride[-1]
                        for i in range(len(whereStride)-1,0,-1):
                            if(start - whereStride[i] != whereStride[i] - whereStride[i-1]):
                                del model[start-1]
                                for j in range(i,len(whereStride)):
                                    whereStride[j] -= 1
                                break
                            else:
                                start = whereStride[i]
            else:
                return createRandomModel(version)
        except:
            traceback.print_exc() 
            return createRandomModel(version)
        
        

def findStride(number:int, depth:int)-> list:
    """Gives where to do the strides so that depthi <= depthi+1

    Args:
        number (int): number of strides to do
        depth (int): the Depth of the Model

    Returns:
        list: a list with the indexe
    """
    whereStride = []
    initStride = 0
    blockLength = 0
    for i in range(number):
        theLength = np.random.randint(initStride, depth//(number+1) + 1) 
        initStride = theLength
        whereStride.append(theLength + blockLength)
        blockLength += initStride
    return whereStride

def testModel(modelLst:list,start:int)->float:
    """Test the model time

    Args:
        modelLst (list): the list to create a model
        start (int): the number of start filters

    Returns:
        float: the time to train for 5 epochs
    """
    testModel = ModelParse(modelLst,start).to("cuda:0")
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(testModel.parameters(), lr=0.001)
    
    
    start_time = time.time()
    for i in range(20):
        
        opt.zero_grad()
        inp = torch.randn(100,3,32,32 , requires_grad=True, device = "cuda:0")
        target = torch.empty(100, dtype=torch.long, device = "cuda:0").random_(100)
        out = testModel(inp)
        ls = loss(out,target)
        ls.backward()
        opt.step()
    return (time.time() - start_time)/30

def trainModel(ModelDict:dict)->dict:
    """Train the Model and evalute the Model

    Args:
        ModelDict (dict): the dict with all the Parameters

    Returns:
        dict: the dict with all the Parameters + loss
    """
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.3, 0.3, 0.3))])
    trainset  = torchvision.datasets.CIFAR100("./data", transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset , batch_size = 100, shuffle = True, num_workers = 3)
    testset = torchvision.datasets.CIFAR100("./data", train= False, transform=transform)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=100,
                                            shuffle=False, num_workers=3)
    
    Net = ModelParse(ModelDict["model"],ModelDict["startFilter"] ).to("cuda:0")
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(Net.parameters(), lr=0.001, momentum = 0.9)
    trainLoss = []
    #klassik learning steps
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")
            opt.zero_grad()
            outputs = Net(inputs)
            ls = loss(outputs,labels)
            ls.backward()
            
            opt.step()

            running_loss += ls.item()
        trainLoss.append(running_loss/len(trainLoader))
    Net.eval()
    valLoss = 0.0
    for i, data in enumerate(testLoader, 0):
        inputs = inputs.to("cuda:0")
        labels = labels.to("cuda:0")
        outputs = Net(inputs)
        ls = loss(outputs,labels)
        valLoss += ls.item()
    
    ModelDict["trainLoss"] = trainLoss
    ModelDict["valLoss"] = valLoss/len(testLoader)
    return ModelDict
