#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform data preprocessing for subsequent
    automatic segmentation

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: July 2023
    """

#%% Imports
# Modules import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
import pandas as pd
from keras import utils
from patchify import patchify
import matplotlib.pyplot as plt
from skimage import io, transform, color
from Utils import Time, SetDirectories, Show
from scipy.stats.distributions import norm, f, t, chi2


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
def PlotDistribution(Data, Variable=''):
        
        # Get data variables
        SortedValues = np.sort(Data)
        N = len(Data)
        X_Bar = np.mean(Data)
        S_X = np.std(Data, ddof=1)

        KernelEstimator = np.zeros(N)
        NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25,0.75]), 0, 1)))
        DataIQR = abs(np.abs(np.quantile(Data,0.75)) - np.abs(np.quantile(Data,0.25)))
        KernelHalfWidth = 0.9*N**(-1/5) * min(S_X,DataIQR/NormalIQR)
        for Value in SortedValues:
            KernelEstimator += norm.pdf(SortedValues-Value,0,KernelHalfWidth*2)
        KernelEstimator = KernelEstimator/N

        # Histogram and density distribution
        TheoreticalDistribution = norm.pdf(SortedValues,X_Bar,S_X)

        Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
        Axes.hist(Data,density=True,bins=20,edgecolor=(0,0,1),color=(1,1,1),label='Histogram')
        Axes.plot(SortedValues,KernelEstimator,color=(1,0,0),label='Kernel Density')
        Axes.plot(SortedValues,TheoreticalDistribution,linestyle='--',color=(0,0,0),label='Normal Distribution')
        plt.xlabel(Variable)
        plt.ylabel('Density (-)')
        plt.legend(loc='upper center',ncol=3,bbox_to_anchor=(0.5,1.15), prop={'size':10})
        plt.show()
def DataStats(Images: list, Plots=False):

    Averages = []
    Stds = []

    # Store average and standard deviation in lab space

    for Image in Images:

        averages = []
        stds = []
        LAB = color.rgb2lab(Image)

        for idx in range(3):
            averages.append(np.mean(LAB[:, :, idx]))
            stds.append(np.std(LAB[:, :, idx]))

        Averages.append(averages)
        Stds.append(stds)

    # Plot distribution
    if Plots:
        PlotDistribution(np.array(Averages)[:,0],'L')
        PlotDistribution(np.array(Averages)[:,1],'A')
        PlotDistribution(np.array(Averages)[:,2],'B')

        PlotDistribution(np.array(Stds)[:,0],'L')
        PlotDistribution(np.array(Stds)[:,1],'A')
        PlotDistribution(np.array(Stds)[:,2],'B')

    aL = norm.fit(np.array(Averages)[:,0])
    aA = norm.fit(np.array(Averages)[:,1])
    aB = norm.fit(np.array(Averages)[:,2])

    sL = norm.fit(np.array(Stds)[:,0])
    sA = norm.fit(np.array(Stds)[:,1])
    sB = norm.fit(np.array(Stds)[:,2])
            
    return np.array([aL, aA, aB]), np.array([sL, sA, sB])
def StainAugmentation(Images, aLAB, sLAB, N):

    """
    According to
        Shen et al. 2022
        RandStainNA: Learning Stain-Agnostic Features from Histology
        Slides by Bridging Stain Augmentation and Normalization
    """

    # Convert image into LAB space
    LAB = []
    S = Images[0].shape
    Time.Process(1,'Convert to LAB')
    for Image in Images:
        lab = color.rgb2lab(Image / 255)
        Nlab = []
        for _ in range(N):
            Nlab.append(lab)
        LAB.append(Nlab)
        Time.Update(len(LAB)/3/len(Images))
    LAB = np.array(LAB)

    # Define random transform parameters (reshape first to match data shape)
    Time.Update(1/3, 'Tranform Images')
    aLAB = np.tile(aLAB.T, N).reshape((2, N, 3), order='A')
    aLAB = np.tile(aLAB, len(Images)).reshape((2, len(Images), N, 3))
    sLAB = np.tile(sLAB.T, N).reshape((2, N, 3), order='A')
    sLAB = np.tile(sLAB, len(Images)).reshape((2, len(Images), N, 3))
    Mean = np.random.normal(aLAB[0], aLAB[1])
    Std = np.random.normal(sLAB[0], sLAB[1])

    # Normalize images according to random templates
    S = LAB.shape
    Norm = np.zeros(S)
    X_Bar = np.mean(LAB, axis=(2,3))
    S_X = np.std(LAB, axis=(2,3), ddof=1)

    X_Bar = np.tile(X_Bar, S[2]*S[3]).reshape(S)
    S_X = np.tile(S_X, S[2]*S[3]).reshape(S)
    Mean = np.tile(Mean, S[2]*S[3]).reshape(S)
    Std = np.tile(Std, S[2]*S[3]).reshape(S)

    Norm = (LAB - X_Bar) / S_X * Std + Mean

    # Convert back to RGB space
    Results = []
    Time.Update(2/3, 'Convert to RGB')
    for i in range(len(Images)):
        results = []
        for j in range(N):
            RGB = color.lab2rgb(Norm[i,j])
            results.append(RGB)
        Results.append(results)
        Time.Update(2/3 + i/3/len(Images))
    Results = np.array(Results) * 255

    Time.Process(0)

    return np.round(Results).astype('uint8')
def DataAugmentation(Image,Label,N):

    Size = Image.shape[:-1]

    # Scale data range between 0 and 1
    Image = Image / 255

    Data = [Image]
    Labels = [Label]

    for iN in range(N-1):

        # Random rotation
        Rot = np.random.randint(0, 360)
        rImage = transform.rotate(Image, Rot)
        rLabel = transform.rotate(Label, Rot, order=0, preserve_range=True)

        # Random flip
        Flip = np.random.binomial(1, 0.5, 2)
        if sum(Flip) == 0:
            fImage = rImage
            fLabel = rLabel
        if Flip[0] == 1:
            fImage = rImage[::-1, :, :]
            fLabel = rLabel[::-1, :]
        if Flip[1] == 1:
            fImage = rImage[:, ::-1, :]
            fLabel = rLabel[:, ::-1]

        # Random translation
        MaxX = int(Size[1] * 0.1)
        MaxY = int(Size[0] * 0.1)

        pImage = np.pad(fImage, ((MaxX, MaxX), (MaxY, MaxY), (0, 0)))
        pLabel = np.pad(fLabel, ((MaxX, MaxY), (MaxX, MaxY)))

        X1 = np.random.randint(0, 2*MaxX)
        Y1 = np.random.randint(0, 2*MaxY)

        X2, Y2 = X1 + Size[1], Y1 + Size[0]
        cImage = pImage[Y1:Y2,X1:X2]
        cLabel = pLabel[Y1:Y2,X1:X2]

        Data.append(cImage)
        Labels.append(cLabel)

    Data = np.array(Data) * 255
    Data = np.round(Data).astype('uint8')

    Labels = np.array(Labels) * 255
    Labels = np.round(Labels).astype('uint8')

    return Data, Labels

#%% Main
# Main part

def Main():

    # Set directory and data
    WD, DD, SD, RD = SetDirectories('FEXHIP-Histology')
    Directory = DD / '03_ManualSegmentation'
    Masks, Images, Data = CollectMasks(Directory)

    # Store original images
    ResultsDir = RD / '02_Preprocessing' / 'Original'
    os.makedirs(ResultsDir, exist_ok=True)
    for i, I in enumerate(Images):
        IName = str(ResultsDir / str('ROI_' + '%02d' % (i) + '.png'))
        io.imsave(IName, I.astype('uint8'))

    ResultsDir = RD / '03_Segmentation' / 'Manual'
    os.makedirs(ResultsDir, exist_ok=True)
    for i, M in enumerate(Masks):
        IName = str(ResultsDir / str('Seg_' + '%02d' % (i) + '.png'))
        Label = np.round(utils.to_categorical(M)[:,:,1:] * 255).astype('int')
        io.imsave(IName, Label.astype('uint8'))

    # Stain augmentation
    N = 2
    AverageLAB, StdLAB = DataStats(Images)
    StainNA = StainAugmentation(Images, AverageLAB, StdLAB, N)

    # Store stain normalized/augmented images
    ResultsDir = RD / '02_Preprocessing' / 'StainNA'
    os.makedirs(ResultsDir, exist_ok=True)
    for i, S in enumerate(StainNA):
        ResultsDir = RD / '02_Preprocessing' / 'StainNA' / str('ROI_' + '%02d' % (i))
        os.makedirs(ResultsDir, exist_ok=True)
        for j, s in enumerate(S):
            IName = str(ResultsDir / str('Stain_' + '%1d' % (j) + '.png'))
            io.imsave(IName, s.astype('uint8'))

    # Perform data augmentation
    N = 8
    ROISize = 256
    Time.Process(1,'Augment Data')
    ResultsDir = RD / '02_Preprocessing' / 'Augmentation'
    os.makedirs(ResultsDir, exist_ok=True)
    Size = StainNA.shape[0] * StainNA.shape[1]
    for s, S in enumerate(StainNA):
        ResultsDir = RD / '02_Preprocessing' / 'Augmentation' / str('ROI_' + '%02d' % (s))
        os.makedirs(ResultsDir, exist_ok=True)
        for i, I in enumerate(S):
            for n in range(N):

                # Flip image
                if np.mod(n,2)-1:
                    fImage = I
                    fLabel = Masks[s]
                elif np.mod(n,4)-1:
                    fImage = I[::-1, :, :]
                    fLabel = Masks[s][::-1, :]
                else:
                    fImage = I[:, ::-1, :]
                    fLabel = Masks[s][:, ::-1]

                # Rotate image of 90 degrees
                if n < 2:
                    rImage = fImage
                    rLabel = fLabel
                elif n < 4:
                    rImage = np.rot90(fImage,1)
                    rLabel = np.rot90(fLabel,1)
                elif n < 6:
                    rImage = np.rot90(fImage,2)
                    rLabel = np.rot90(fLabel,2)
                else:
                    rImage = np.rot90(fImage,3)
                    rLabel = np.rot90(fLabel,3)

                # Select random location
                Min = 0
                Max = rImage.shape[0] - ROISize
                X0, Y0 = np.random.randint(Min, Max, 2)
                ROI = rImage[X0:X0+ROISize, Y0:Y0+ROISize]
                Lab = rLabel[X0:X0+ROISize, Y0:Y0+ROISize].astype('uint8')

                # Save transformed image
                io.imsave(str(ResultsDir / ('Image_' + '%1d' % (i) + '%1d' % (n) + '.png')), ROI)
                io.imsave(str(ResultsDir / ('Label_' + '%1d' % (i) + '%1d' % (n) + '.png')), Lab, check_contrast=False)
        Time.Update((s+1) * (i+1) / Size)

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

