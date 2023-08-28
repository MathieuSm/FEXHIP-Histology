#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script analyses the manual segmentation data

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: June 2023
    """

#%% Imports
# Modules import

import os
import argparse
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from skimage import morphology, measure
from matplotlib.colors import ListedColormap
from Utils import Time, SetDirectories, Show


#%% Functions
# Define functions

def CollectMasks(Directory):

    Time.Process(1,'Collect masks')

    Masks = []
    Names = []
    Files = []
    Images = []
    Folders = [F for F in os.listdir(Directory) if os.path.isdir(Directory / F)]
    for iF, Folder in enumerate(Folders):
        Path = Directory / Folder / 'segmentation'
        SubFolders = os.listdir(Path)
        if 'SegmentationClass' in SubFolders:
            SubPath = Path / 'SegmentationClass'
            ImageFiles = os.listdir(SubPath)
            for File in ImageFiles:
                CMask = io.imread(str(SubPath / File))
                Colors = np.unique(CMask.reshape(-1, CMask.shape[2]), axis=0)
                Mask = np.zeros((CMask.shape[0], CMask.shape[1], len(Colors)))
                for iC, C in enumerate(Colors):
                    F1 = (CMask[:,:,0] == C[0]) * 1
                    F2 = (CMask[:,:,1] == C[1]) * 1
                    F3 = (CMask[:,:,2] == C[2]) * 1
                    Mask[:,:,iC] = F1 * F2 * F3
                Masks.append(Mask)
                Names.append(Folder)
                Files.append(File)
                Images.append(io.imread(str(Directory / Folder / 'data' / File)))
        
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
    Directory = DD / '03_ManualSegmentation'
    Masks, Images, Data = CollectMasks(Directory)

    # Get common ROI
    C = ['Sample','Side','Quadrant','ROI']
    Sample = ['418', 'Left', 'Inferior','4']
    Idx = Data[Data[C] == Sample][C].dropna().index

    # Show segmentations for common ROI
    Figure, Axis = plt.subplots(3,4, dpi=500)
    for i in range(3):
        for j in range(4):
            Axis[i,j].set_title(Data.loc[Idx[i*4+j],'Operator'])
            Axis[i,j].imshow(Masks[Idx[i*4+j]][:,:,1:])
            Axis[i,j].axis('off')
    plt.show(Figure)

    # Segments canals mean areas
    Mask = np.zeros(Masks[Idx[0]].shape)
    Osteocytes = pd.DataFrame(columns=['Operator','Object','Area'])
    HCanals = pd.DataFrame(columns=['Operator','Object','Area'])
    CMLines = pd.DataFrame(columns=['Operator','Object','Area'])
    LineThickness = morphology.disk(2)
    Ol, Hl, Cl = 0, 0, 0
    for i, idx in enumerate(Idx):
        iMask = Masks[idx]
        Mask += iMask

        Labels = morphology.label(iMask[:,:,1])
        Props = measure.regionprops(Labels, Images[idx])
        for iP, P in enumerate(Props):
            Osteocytes.loc[Ol] = [Data.loc[Idx[i],'Operator'],iP+1,P.area]
            Ol += 1

        Labels = morphology.label(iMask[:,:,2])
        Props = measure.regionprops(Labels, Images[idx])
        for iP, P in enumerate(Props):
            HCanals.loc[Hl] = [Data.loc[Idx[i],'Operator'],iP+1,P.area]
            Hl += 1

        Erode = morphology.binary_erosion(iMask[:,:,3], morphology.disk(1))
        Skeleton = morphology.skeletonize(Erode)
        CMLine = morphology.binary_dilation(Skeleton, LineThickness)
        Mask[:,:,3] -= iMask[:,:,3]
        Mask[:,:,3] += CMLine
        Labels = morphology.label(CMLine)
        Props = measure.regionprops(Labels, Images[idx])
        for iP, P in enumerate(Props):
            CMLines.loc[Cl] = [Data.loc[Idx[i],'Operator'],iP+1,P.area]
            Cl += 1

    Mask = Mask / np.max(Mask, axis=(0,1))

    # Get Variability
    Ov, Oc = np.unique(Mask[:,:,1], return_counts=True)
    Hv, Hc = np.unique(Mask[:,:,2], return_counts=True)
    Cv, Cc = np.unique(Mask[:,:,3], return_counts=True)

    v = []
    for a in [Oc, Ov, Hc, Hv, Cc, Cv]:
        v.append(a[:0:-1])
    Oc, Ov, Hc, Hv, Cc, Cv = v

    Figure, Axis = plt.subplots(1,3, sharex=False, sharey=True, figsize=(10,3.5))
    Axis[0].bar(Ov, np.cumsum(Oc) / sum(Oc), width=Ov[-1], edgecolor=(1,0,0), color=(1, 1, 1, 0), label='Osteocytes')
    Axis[1].bar(Hv, np.cumsum(Hc) / sum(Hc), width=Hv[-1], edgecolor=(0,1,0), color=(1, 1, 1, 0), label='Haversian Canals')
    Axis[2].bar(Cv, np.cumsum(Cc) / sum(Cc), width=Cv[-1], edgecolor=(0,0,1), color=(1, 1, 1, 0), label='Cement Lines')
    for i in range(3):
        Axis[i].set_xlim([1.1,0])
        Axis[i].legend(loc='upper left')
    Axis[1].set_xlabel('Agreement Ratio (-)')
    Axis[0].set_ylabel('Relative Cumulative Count (-)')
    plt.show()

    # Plot mask data
    for Areas in [Osteocytes, HCanals]:
        Operators = [O for O in Areas['Operator'].unique()]
        Boxes = [Areas[Areas['Operator']==O]['Area'] for O in Operators]
        Show.BoxPlot(Boxes, Labels=['Operator','Area (px)'])
        Show.BoxPlot([Areas['Area']])

    Numbers = [Osteocytes[Osteocytes['Operator']==O]['Object'].max() for O in Operators]
    Show.BoxPlot([Numbers])

    CMAreas = [df['Area'].sum() for idx, df in CMLines.groupby(by='Operator')]
    Show.BoxPlot([CMAreas])

    # Plot average segmentation
    N = 256
    CValues = np.zeros((N, 4))
    CValues[:, 0] = np.linspace(0, 1, N)
    CValues[:, 1] = np.linspace(1, 0, N)
    CValues[:, 2] = np.linspace(1, 0, N)
    CValues[:, -1] = np.linspace(1.0, 1.0, N)
    CMP = ListedColormap(CValues)

    NanMask = Mask.copy()
    NanMask[Mask == 0.0] = np.nan
    
    Figure, Axis = plt.subplots(1,3, dpi=500)
    for i in range(3):
        Plot = Axis[i].imshow(NanMask[:,:,i+1], cmap=CMP)
        Axis[i].imshow(Images[idx])
        Axis[i].axis('off')
    CBarAxis = Figure.add_axes([0.2, 0.25, 0.6, 0.025])
    plt.colorbar(Plot, cax=CBarAxis, orientation='horizontal', label='Segmentation ratio (-)')
    plt.show(Figure)

    Figure, Axis = plt.subplots(1,3, dpi=500)
    for i in range(3):
        Axis[i].imshow(Images[idx])
        Plot = Axis[i].imshow(NanMask[:,:,i+1], cmap=CMP)
        Axis[i].axis('off')
    CBarAxis = Figure.add_axes([0.2, 0.25, 0.6, 0.025])
    plt.colorbar(Plot, cax=CBarAxis, orientation='horizontal', label='Segmentation ratio (-)')
    plt.show(Figure)


    Cols = ['Osteocytes','Haversian Canal','Cement Line']
    Scores = pd.DataFrame(index=Data['Operator'].unique(), columns=Cols)
    Scores.index.name = 'Operator'
    for idx in Idx:
        Operator = Data.loc[idx, 'Operator']
        for i in range(3):
            Scores.loc[Operator,Cols[i]] = np.nanmean(NanMask[:,:,i+1] * Masks[idx][:,:,i+1])
    Scores = Scores.dropna()
    Scores = Scores / Scores.max(axis=0)
    Scores.to_csv(str(RD / '02_Scores.csv'))

    return


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
