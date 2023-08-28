#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script compare automatic segmentation using random forest and Unet

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

def ConfusionMatrix(Prediction, Truth, Title=None):
    CM = metrics.confusion_matrix(Truth,Prediction,normalize=None)
    CM2 = metrics.confusion_matrix(Truth,Prediction,normalize='true')
    CM3 = metrics.confusion_matrix(Truth,Prediction,normalize='pred')
    VSpace = 0.2
    Ticks = ['Osteocytes', 'Haversian\nCanals', 'Cement\nLine']
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
    if Title:
        Axis.set_title(Title)
    Axis.set_xlabel('Ground Truth',color=(0,0,1))
    Axis.set_ylabel('Predictions',color=(1,0,0))
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

    # Read segmentation results
    UNet = []
    RF = []
    UNet_RF = []
    Ref = []
    UDir = ResultsDir / 'Unet'
    FDir = ResultsDir / 'RandomForest'
    UFDir = ResultsDir / 'UNet_RF'
    MDir = ResultsDir / 'Manual'
    Files = os.listdir(UDir)
    Files.sort()
    for F in Files:
        Seg = io.imread(str(UDir / F))
        UNet.append(Seg)
        Seg = io.imread(str(FDir / F))
        RF.append(Seg)
        Seg = io.imread(str(UFDir / F))
        UNet_RF.append(Seg)
        Seg = io.imread(str(MDir / F))
        Ref.append(Seg)

    UNet = np.array(UNet)
    RF = np.array(RF)
    UNet_RF = np.array(UNet_RF)
    Ref = np.array(Ref)

    # Look at random testing image
    Random = np.random.randint(0, len(Ref)-1)

    Figure, Axis = plt.subplots(2,3)
    Axis[0,0].imshow(Images[Random])
    Axis[0,1].imshow(Ref[Random])
    Axis[0,2].imshow(RF[Random])
    Axis[1,1].imshow(UNet[Random])
    Axis[1,2].imshow(UNet_RF[Random])
    for i in range(2):
        for j in range(3):
            Axis[i,j].axis('off')
    plt.tight_layout()
    plt.show()

    # Assess model
    Truth = np.argmax(Ref, axis=-1).ravel()
    Prediction = np.argmax(RF, axis=-1).ravel()
    ConfusionMatrix(Prediction, Truth, Title='Random Forest')
    Prediction = np.argmax(UNet, axis=-1).ravel()
    ConfusionMatrix(Prediction, Truth, Title='U-net')
    Prediction = np.argmax(UNet_RF, axis=-1).ravel()
    ConfusionMatrix(Prediction, Truth, Title='U-net + RF')


