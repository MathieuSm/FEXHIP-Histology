#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform automatic segmentation using random forest

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: June 2023
    """

#%% Imports
# Modules import

import os
import joblib
import argparse
import numpy as np
import pandas as pd
from skimage import io
from keras import utils
from sklearn import metrics
from skimage import feature
from patchify import patchify
import matplotlib.pyplot as plt
from Utils import Time, SetDirectories
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#%% Functions
# Define functions

def CollectMasks(Directory):

    Time.Process(1,'Collect masks')

    Masks = []
    Names = []
    Files = []
    Images = []
    Folders = [F for F in os.listdir(Directory) if os.path.isdir(Directory / F)]
    Folders.sort()
    for iF, Folder in enumerate(Folders):
        Path = Directory / Folder / 'segmentation'
        SubFolders = os.listdir(Path)
        SubFolders.sort()
        if 'SegmentationClass' in SubFolders:
            SubPath = Path / 'SegmentationClass'
            ImageFiles = os.listdir(SubPath)
            ImageFiles.sort()
            for File in ImageFiles:
                CMask = io.imread(str(SubPath / File))
                Colors = np.unique(CMask.reshape(-1, CMask.shape[2]), axis=0)
                Mask = np.zeros(CMask.shape[:-1])
                for iC, C in enumerate(Colors):
                    F1 = (CMask[:,:,0] == C[0]) * 1
                    F2 = (CMask[:,:,1] == C[1]) * 1
                    F3 = (CMask[:,:,2] == C[2]) * 1
                    Mask += (F1 * F2 * F3) * iC
                Masks.append(Mask)
                Names.append(Folder)
                Files.append(File)

                Image = io.imread(str(Directory / Folder / 'data' / File))
                Images.append(Image[:,:,:-1])
        
        Time.Update((iF+1)/len(Folders))

    Columns = ['Operator', 'Sample', 'Side', 'Quadrant', 'ROI']
    Data = pd.DataFrame(columns=Columns)
    Data[Columns[0]] = Names
    Data[Columns[1]] = [S[:3] for S in Files]
    Data[Columns[2]] = [S[3] for S in Files]
    Data[Columns[2]] = Data[Columns[2]].replace({'L':'Left','R':'Right'})
    Data[Columns[3]] = [S[4] for S in Files]
    Data[Columns[3]] = Data[Columns[3]].replace({'L':'Superior','M':'Inferior'})
    Data[Columns[4]] = [S[6] for S in Files]

    Time.Process(0)

    return Masks, Images, Data

#%% Main
# Main part

def Main():

    # Set directory and data
    WD, DD, SD, RD = SetDirectories('FEXHIP-Histology')
    ResultsDir = RD / '03_Segmentation'
    os.makedirs(ResultsDir, exist_ok=True)
    Directory = DD / '03_ManualSegmentation'
    Masks, Images, Data = CollectMasks(Directory)

    # Set sample weights according to manual segmentation
    Scores = pd.read_csv(str(RD / '02_Scores.csv'), index_col=[0])
    sWeights = np.zeros(np.array(Masks).shape)
    for i, Op in enumerate(Data['Operator']):
        if Op in Scores.index:
            Score = Scores[Scores.index == Op].values[0]
        else:
            Score = Scores.mean().values
        BG = np.where(Masks[i] == 0)
        OS = np.where(Masks[i] == 1)
        HC = np.where(Masks[i] == 2)
        CL = np.where(Masks[i] == 3)
        sWeights[i][BG] = Score.mean()
        sWeights[i][OS] = Score[0]
        sWeights[i][HC] = Score[1]
        sWeights[i][CL] = Score[2]

    # Decompose images into patch of a given size
    Size = 256
    ROIs = []
    Labels = []
    Weights = []
    Time.Process(1,'Patchify images')
    for i, I in enumerate(Images):

        # Compute patch size
        NPatches = np.ceil(np.array(I.shape)[:2] / np.array((Size, Size))).astype(int)
        Overlap = NPatches * np.array((Size, Size)) - np.array(I.shape)[:2]
        Step = np.append(Size - Overlap, [3])
    
        # Generate patches
        IPatches = patchify(I, (Size, Size, 3), step=Step)
        MPatches = patchify(Masks[i], (Size, Size), step=Step[:-1])
        WPatches = patchify(sWeights[i], (Size, Size), step=Step[:-1])

        # Store resulting images
        for j in range(NPatches[0]):
            for k in range(NPatches[1]):
                ROIs.append(IPatches[j,k,0])
                Labels.append(MPatches[j,k])
                Weights.append(WPatches[j,k])

        Time.Update((i+1) / len(Images))
    Time.Process(0)

    ROIs = np.array(ROIs)
    Labels = np.array(Labels)
    Weights = np.array(Weights)

    # Extract features
    Features = []
    for ROI in ROIs:
        Feature = feature.multiscale_basic_features(ROI, channel_axis=-1, sigma_min=2, num_sigma=3)
        Features.append(Feature)
    Features = np.array(Features).reshape(-1, Feature.shape[-1])

    # Split into train and test data
    XTrain, XTest, YTrain, YTest, WTrain, WTest = train_test_split(Features, Labels.reshape(-1), Weights, random_state=42)

    # Artificially balance data for faster fit
    Values, Counts = np.unique(YTrain, return_counts=True)
    Indices = pd.DataFrame(YTrain).groupby(0).sample(min(Counts//2)).index
    XTrain = XTrain[Indices]
    YTrain = YTrain[Indices]
    WTrain = WTrain[Indices]

    # Instanciate and fit random forest classifier
    RFc = RandomForestClassifier(n_estimators=100,
                                 oob_score=True,
                                 n_jobs=-1,
                                 verbose=2,
                                 class_weight='balanced')
    RFc.fit(XTrain, YTrain+1, sample_weight=WTrain)
    joblib.dump(RFc, str(ResultsDir / 'RandomForest_Compressed.joblib'), compress=True)

    Results = RFc.predict(XTest)

    # Assess model
    Truth = YTest + 1
    Prediction = Results
    CM = metrics.confusion_matrix(Truth,Prediction,normalize=None)
    CM2 = metrics.confusion_matrix(Truth,Prediction,normalize='true')
    CM3 = metrics.confusion_matrix(Truth,Prediction,normalize='pred')
    VSpace = 0.2
    Ticks = ['IT', 'O', 'HC', 'CL']
    Figure, Axis = plt.subplots(1,1, figsize=(5.5,4.5))
    Axis.matshow(CM3, cmap='binary', alpha=0.33)
    for Row in range(CM.shape[0]):
        for Column in range(CM.shape[1]):
            Axis.text(x=Row, y=Column, position=(Row,Column), va='center', ha='center', s=CM[Row, Column])
            Axis.text(x=Row, y=Column, position=(Row,Column+VSpace), va='center', ha='center', s=round(CM2[Row, Column],2), color=(0,0,1))
            Axis.text(x=Row, y=Column, position=(Row,Column-VSpace), va='center', ha='center', s=round(CM3[Row, Column],2), color=(1,0,0))
    Axis.xaxis.set_ticks_position('bottom')
    Axis.set_xticks(np.arange(len(Ticks)),Ticks)
    Axis.set_yticks(np.arange(len(Ticks)),Ticks)
    Axis.set_ylim([-0.49,CM.shape[0]-0.5])
    Axis.set_title('Total: ' + str(Truth.size))
    Axis.set_xlabel('Ground Truth',color=(0,0,1))
    Axis.set_ylabel('Predictions',color=(1,0,0))
    plt.show()

    # Look at random testing image
    Random = np.random.randint(0, len(ROIs)-1)
    TestImage = ROIs[Random]
    TestLabel = Labels[Random]
    Features = feature.multiscale_basic_features(TestImage, channel_axis=-1, sigma_min=2, num_sigma=3)
    S = Features.shape
    Prediction = RFc.predict(Features.reshape((S[0]*S[1], S[-1])))
    Prediction = Prediction.reshape((S[0],S[1]))

    Figure, Axis = plt.subplots(1,3)
    Axis[0].imshow(TestImage)
    Axis[1].imshow(TestLabel)
    Axis[2].imshow(Prediction)
    for i in range(3):
        Axis[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Collect random forest predictions and save
    PredDir = ResultsDir / 'RandomForest'
    os.makedirs(PredDir, exist_ok=True)
    for i, I in enumerate(Images):

        Features = feature.multiscale_basic_features(I, channel_axis=-1, sigma_min=2, num_sigma=3)
        S = Features.shape
        Prediction = RFc.predict(Features.reshape((S[0]*S[1], S[-1])))
        Prediction = Prediction.reshape((S[0],S[1]))
        Seg = utils.to_categorical(Prediction)[:,:,1:] * 255

        IName = str(PredDir / str('Seg_' + '%02d' % (i) + '.png'))
        io.imsave(IName, Seg[:,:,1:].astype('uint8'))



#%% If main
if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main()
# %%
