import lutorpy as lua
import numpy as np
import scipy.io as sio
from extract_acoustic_feature import *

def extract_hidden_features(inFile,outFile,inModel)
    """ Load the music file
        Extract CF hidden features from the song given
        :return realOutput: a vector contains hidden features
        :Author: Chris Hu
    """
    require ("torch")
    require ("cutorch")
    require ("nn")
    require ("cunn")
    require ("dp")

    #inModel = 'log/cnn160801172442.dat'
    #inFile = '/home/share/MillionSongSubset/download/SOAAEHR12A6D4FB060.mp3'
    #outFile = 'testtemp.mat'

    features = extract_acoustic_feature(inFile)
    cnn_features = np.zeros((1,1,128,999))
    cnn_features[0, 0, :, :] = features[:, :999]
    testData = torch.fromNumpyArray(cnn_features)

    model = torch.load(inModel)

    cuTestData = testData._cuda()
    cuModel = model._cuda()
    cuModel._evaluate()

    cuRealOutput = cuModel._forward(cuTestData)

    realOutput = cuRealOutput.asNumpyArray()

    sio.savemat(outFile,{'x':realOutput})

    return realOutput
