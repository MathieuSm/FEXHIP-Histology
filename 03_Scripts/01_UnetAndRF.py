#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform automatic segmentation using unet and random forest

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: June 2023
    """

#%% Imports
# Modules import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import argparse
import numpy as np
import pandas as pd
from keras import utils
from sklearn import metrics
from patchify import patchify
import matplotlib.pyplot as plt
from keras.models import load_model
from Utils import Time, SetDirectories
from sklearn.model_selection import KFold
from skimage import io, feature, morphology
from sklearn.ensemble import RandomForestClassifier


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

def ShowKFolds(Results, Sites):

    NSplits = len(Results)

    for Metric in ['Precision', 'Recall']:
        
        Figure, Axis = plt.subplots(2,2, sharex=True, sharey=True)
        for i in range(2):
            for j in range(2):
                # Plot bars
                Axis[i,j].bar(np.arange(NSplits),
                            Results[Metric][Sites[i*2+j]],
                            edgecolor=(0,0,1), color=(1, 1, 1, 0))

                # Plot mean value
                Mean = Results[Metric][Sites[i*2+j]].mean()
                Axis[i,j].plot(np.arange(NSplits),
                            np.repeat(Mean, NSplits),
                            color=(1,0,0))
                
                # Plot standard deviation range
                Std = Results[Metric][Sites[i*2+j]].std()
                Axis[i,j].fill_between(np.arange(NSplits),
                                    np.repeat(Mean - Std, NSplits),
                                    np.repeat(Mean + Std, NSplits),
                                    color=(0,0,0,0.2))
                
                # Set axis label and title
                Axis[i,j].set_title(Sites[i*2+j])

                Axis[1,j].set_xlabel('Folds')
                Axis[i,0].set_ylabel(Metric)
        plt.show()

    return

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

    # Load Unet and create osteocytes mask
    UNet = load_model(str(ResultsDir / 'UNet.hdf5'))
    Osteocytes = []
    Time.Process(1,'Segment osteocytes')
    for i, ROI in enumerate(ROIs):
        Mask = UNet.predict(np.expand_dims(ROI / 255, 0), verbose=0)[0]
        Osteocytes.append(Mask[:,:,1] > 0.5)
        Time.Update(i / len(ROIs))

    Osteocytes = np.array(Osteocytes)
    Time.Process(0)

    # Extract features for random forest fit
    Features = []
    Time.Process(1,'Extract features')
    for i, ROI in enumerate(ROIs):
        Feature = feature.multiscale_basic_features(ROI, channel_axis=-1, sigma_min=2, num_sigma=3)
        Features.append(Feature)
        Time.Update(i / len(ROIs))

    Features = np.array(Features)
    Features = Features.reshape(-1, Features.shape[-1])
    Time.Process(0)
    
    # Balance and subsample data for faster fit
    Values, Counts = np.unique(Labels, return_counts=True)
    Indices = pd.DataFrame(Labels.ravel()).groupby(0).sample(min(Counts//2)).index
    SubFeatures = Features[Indices]
    SubLabels = Labels.ravel()[Indices]
    Mask = Osteocytes.ravel()[Indices]
    sWeights = Weights.ravel()[Indices]

    # Instanciate random forest classifier
    RFc = RandomForestClassifier(n_estimators=100,
                                oob_score=True,
                                n_jobs=-1,
                                verbose=0,
                                class_weight='balanced')

    # Use KFold cross validation
    NSplits = 7
    CV = KFold(n_splits=NSplits, shuffle=True, random_state=42)
    FoldsScores = []
    Time.Process(1, 'KFold validation')
    for i, (Train, Test) in enumerate(CV.split(SubFeatures, SubLabels)):

        # Get train and test data
        XTrain = SubFeatures[Train]
        YTrain = SubLabels[Train]
        WTrain = sWeights[Train]
        XTest = SubFeatures[Test]
        YTest = SubLabels[Test]
        
        # Get osteocytes segmented by Unet
        OTrain = Mask[Train]
        OTest = Mask[Test]

        # Drop pixels corresponding to osteocytes
        SubXTrain = XTrain[~OTrain]
        SubYTrain = YTrain[~OTrain]
        SubWTrain = WTrain[~OTrain]
        SubXTest = XTest[~OTest]
        SubYTest = YTest[~OTest]

        # Fit random forest
        RFc.fit(SubXTrain, SubYTrain+1, sample_weight=SubWTrain)

        # Perform prediction
        Pred = RFc.predict(SubXTest)
        
        # Compute scores
        RF_Precision = metrics.precision_score(Pred, SubYTest+1, average=None)
        UNet_Precision = metrics.precision_score(YTest==1, OTest, average=None)
        RF_Recall = metrics.recall_score(Pred, SubYTest+1, average=None)
        UNet_Recall = metrics.recall_score(YTest==1, OTest, average=None)
        Precision = [RF_Precision[0], UNet_Precision[1], RF_Precision[2], RF_Precision[3]]
        Recall = [RF_Recall[0], UNet_Recall[1], RF_Recall[2], RF_Recall[3]]
        FoldsScores.append([Precision, Recall])

        # Update time
        Time.Update(i / NSplits)
    Time.Process(0)

    # Save scores and plot
    Sites = ['Interstitial Tissue', 'Osteocytes', 'Haversian Canals', 'Cement Lines']
    Cols = pd.MultiIndex.from_product([['Precision','Recall'], Sites])
    FoldsScores = np.array(FoldsScores).reshape(NSplits, len(Cols))
    FoldsScores = pd.DataFrame(FoldsScores, columns=Cols)
    FoldsScores.to_csv(str(ResultsDir / 'FoldsScores.csv'), index=False)

    ShowKFolds(FoldsScores, Sites)

    # Fit random forest classifier with all data and save it
    RFc.verbose = 2
    XTrain = Features[~Osteocytes.ravel()]
    WTrain = Weights.ravel()[~Osteocytes.ravel()]
    YTrain = Labels.ravel()[~Osteocytes.ravel()]
    RFc.fit(XTrain, YTrain+1, sample_weight=WTrain)
    joblib.dump(RFc, str(ResultsDir / 'RandomForest.joblib'))

    # RFc = joblib.load(str(ResultsDir / 'RandomForest.joblib'))

    # Assess overall model
    Truth = Labels.ravel() + 1
    Prediction = RFc.predict(Features)
    Prediction[Osteocytes.ravel()] = 2
    CM = metrics.confusion_matrix(Truth,Prediction,normalize=None)
    CM2 = metrics.confusion_matrix(Truth,Prediction,normalize='true')
    CM3 = metrics.confusion_matrix(Truth,Prediction,normalize='pred')
    VSpace = 0.2
    Ticks = ['Interstitial\nTissue', 'Osteocytes', 'Haversian\nCanals', 'Cement\nLine']
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

    # Save predictions
    PredDir = ResultsDir / 'Automatic'
    os.makedirs(PredDir, exist_ok=True)
    ROIsPred = Prediction.reshape(ROIs.shape[:-1])
    for i, I in enumerate(Images):

        S = I.shape
        Seg = np.zeros((S[0],S[1],4))
        for j in range(2):
            for k in range(2):
                Pred = ROIsPred[4*i+j*2+k]
                Sp = Pred.shape
                Categories = utils.to_categorical(Pred)
                Step = S[0] - Sp[0]
                X0 = j*Step
                Y0 = k*Step
                Seg[X0:X0+Sp[0], Y0:Y0+Sp[1],:-1] = Categories[:,:,-3:] * 255
                BG = Categories[:,:,1].astype('bool')
                Seg[X0:X0+Sp[0], Y0:Y0+Sp[1],-1][~BG] = 255
                Seg[:,:,1][Seg[:,:,2] == 255] = 255

        IName = str(PredDir / str('Seg_' + '%02d' % (i) + '.png'))
        Figure, Axis = plt.subplots(1,1)
        Axis.imshow(Images[i])
        Axis.imshow(Seg, alpha=0.5)
        Axis.axis('off')
        plt.subplots_adjust(0,0,1,1)
        plt.savefig(IName)


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
