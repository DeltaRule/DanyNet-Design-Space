from GeneralUtils import createRandomModel,trainModel
from makeModel import ModelParse
import pickle
from tqdm import tqdm
VERSION = 0 
LOAD = True
SPAWN = 500
if __name__ == "__main__":
    #Version 0
    #spawn Models
    spawnedModels = []
    #create Model
    if(LOAD):
        outfile = open(f"./prespawned/version{VERSION}.pkl",'rb')
        spawnedModels = pickle.load(outfile)
        outfile.close()
    else:
        for _ in tqdm(range(SPAWN)):
            spawnedModels.append(createRandomModel(version = VERSION))
        outfile = open("./prespawned/version{VERSION}.pkl",'wb')
        pickle.dump(spawnedModels,outfile)
        outfile.close()
    #Learn Model
    for i in tqdm(range(len(spawnedModels))):
        spawnedModels[i] = trainModel(spawnedModels[i])
    outfile = open("./prespawned/version{VERSION}_trained.pkl",'wb')
    pickle.dump(spawnedModels,outfile)
    outfile.close()

