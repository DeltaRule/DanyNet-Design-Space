from GeneralUtils import createRandomModel,trainModel,searchForInsert,empiricalBootstrap
from makeModel import ModelParse
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

VERSION = 0 
LOAD = True
SPAWN = 500
if __name__ == "__main__":
    #Version 0
    #spawn Models
    spawnedModels = []
    #create Model
    if(LOAD):
        outfile = open(f"./prespawned/version{VERSION}_trained.pkl",'rb')
        spawnedModels = pickle.load(outfile)
        outfile.close()
    else:
        for _ in tqdm(range(SPAWN)):
            spawnedModels.append(createRandomModel(version = VERSION))
        outfile = open(f"./prespawned/version{VERSION}.pkl",'wb')
        pickle.dump(spawnedModels,outfile)
        outfile.close()
        #Learn Model
        for i in tqdm(range(len(spawnedModels))):
            spawnedModels[i] = trainModel(spawnedModels[i])
        outfile = open(f"./prespawned/version{VERSION}_trained.pkl",'wb')
        pickle.dump(spawnedModels,outfile)
        outfile.close()
    
    #sort the model after the error for easier Bootstrap and for ploting
    sortModel = {
        "stride":[],
        "kernel3":[],
        "swish":[],
        "sne":[],
        "weightnomalization": [],
        "instancenomalization": [],
        "batchnomalization":[],
        "noGroups": [],
        "Dec First": [],
        "Dec Second": [],
        "depth": [],
        "startFilter": [],
        "Validation error":[],
        "minimal training error":[],
        "model": [],
    }
    
    for model in tqdm(spawnedModels):
        getValError = model.pop("valLoss")
        whereInsert = searchForInsert(sortModel["Validation error"], getValError)
        sortModel["Validation error"].insert(whereInsert,getValError)
        for modelKey in model:
            if(modelKey == "norms"):
                sortModel["weightnomalization"].insert(whereInsert,model[modelKey][0])
                sortModel["instancenomalization"].insert(whereInsert,model[modelKey][1])
                sortModel["batchnomalization"].insert(whereInsert,model[modelKey][2])
            elif(modelKey == "layerType"):
                sortModel["noGroups"].insert(whereInsert,model[modelKey][0])
                sortModel["Dec First"].insert(whereInsert,model[modelKey][1])
                sortModel["Dec Second"].insert(whereInsert,model[modelKey][2])
            elif(modelKey == "trainLoss"):
                sortModel["minimal training error"].insert(whereInsert, min(model[modelKey]))
            else:
                sortModel[modelKey].insert(whereInsert,model[modelKey])

    for key in sortModel:
        if(key != "Validation error" and key != "model"):
            
            plt.title(key)
            plt.xlabel(key)
            plt.ylabel("Validation Loss")
            plt.scatter(sortModel[key], sortModel["Validation error"])
            #it is already sorted so we don't need to give the loss aswell
            plt.axvspan(*empiricalBootstrap(sortModel[key]), alpha = 0.3)
            plt.savefig(f"./outputImages/{key}_Version{VERSION}.png")
            plt.clf()
        elif(key == "Validation error"):
            plt.title("Design Space Evaluation")
            plt.xlabel("Error")
            plt.ylabel("EDF")
            edf = np.linspace(0,1.0,len(sortModel[key]))
            plt.plot(sortModel[key], edf)
            plt.savefig(f"./outputImages/DesignSpaceEDF_Version{VERSION}.png")
            plt.clf()
