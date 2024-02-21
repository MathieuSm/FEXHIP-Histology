#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script performs automatic segmentation of blue stained
    cortical bone using U-net and random forest

    Detailed description:
        Simon et al. (2024)
        Automatic Segmentation of Cortical Bone Structure
        SomeJournal, x(x), xxx-xxx
        https://doi.org/

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: October 2023
    """

#%% Imports
# Modules import

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import joblib
import argparse
import numpy as np
from keras import utils
from pathlib import Path
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage import feature, io, measure, morphology
from Utils import Time, Reference, StainNA, Unet_Probabilities

#%% Functions
# Define functions

def Segmentation(Image:np.array, Unet, RFc) -> np.array:

    # Record elapsed time
    Time.Process(1, 'Segment ROI')

    # Perform stain normalization
    Time.Update(1/5, 'Stain normalization')
    Norm = StainNA(Image, Reference.Mean, Reference.Std)

    # Get probability from Unet
    Time.Update(2/5, 'U-net probabilities')
    Prob = Unet_Probabilities(Unet, Norm)
    UPred = np.argmax(Prob, axis=-1)+1

    # Compute image features for random forest
    Time.Update(3/5, 'Image features')
    Kwargs = {'channel_axis':-1,
              'sigma_min':2.0,
              'num_sigma':3}
    Feat = feature.multiscale_basic_features(Image,**Kwargs)

    # Concatenate Unet probability with image features
    Prob = Prob.reshape(-1, Prob.shape[-1])
    Feat = Feat.reshape(-1, Feat.shape[-1])
    Features = np.concatenate([Feat, Prob], axis=1)

    # Get random forest prediction
    Time.Update(4/5, 'RF prediction')
    RFPred = RFc.predict(Features)
    RFPred = RFPred.reshape(Image.shape[:-1])

    Time.Process(0)

    return RFPred

def RemoveIsland(Mask:np.array, Threshold:int) -> np.array:

    """
    Remove island smaller than a given size
    """

    Regions = measure.label(Mask)
    Values, Counts = np.unique(Regions, return_counts=True)
    Cleaned = np.isin(Regions, Values[1:][Counts[1:] > Threshold])

    return Cleaned

def ReattributePixels(Cleaned:np.array, Mask:np.array, Values:list) -> np.array:

    """
    Modify pixel labels by the given values
    """

    Cleaned[(Cleaned == Values[1])*~Mask] = Values[0]
    Cleaned[(Cleaned == Values[1])* Mask] = Values[1]
    Cleaned[(Cleaned == Values[0])*~Mask] = Values[0]
    Cleaned[(Cleaned == Values[0])* Mask] = Values[1]

    return Cleaned

def CleanSegmentation(Pred:np.array) -> np.array:

    """
    Clean segmentation by connected component thresholding
    and thin cement lines by erosion 
    """

    # Get cement lines, Haversian canals, and osteocytes masks
    Clean = Pred.copy()
    OS = Clean == 2
    HC = Clean == 3
    CL = Clean == 4

    # Keep connected regions with more than 15 pixels
    CleanOS = RemoveIsland(OS, 25)

    # Reattribute pixels labels
    Clean = ReattributePixels(Clean, CleanOS, [1,2])

    # Keep connected regions with more than 200 pixels
    CleanHC = RemoveIsland(HC, 200)

    # Reattribute pixels labels
    Clean = ReattributePixels(Clean, CleanHC, [1,3])

    # Pad array to connect cement lines at the border
    Pad = np.pad(CL+OS, ((1,1),(1,1)), mode='constant', constant_values=True)

    # Keep connected regions with more than 1000 pixels
    CleanCL = RemoveIsland(Pad, 1000)

    # Thin cement lines by erosion
    CleanCL = morphology.binary_erosion(CleanCL, morphology.disk(1))
    CleanCL = CleanCL[1:-1,1:-1] * CL

    # Reattribute pixels labels
    Clean = ReattributePixels(Clean, CleanCL, [1,4])

    return Clean

def SaveSegmentation(I:np.array, S:np.array, FigName:str) -> None:

    """
    Save segmentation mask and overlay of
    original image and segmentation mask
    """

    Categories = utils.to_categorical(S)

    MaskName = FigName.parent / (FigName.name[:-4] + '_Mask.png')

    FigSize = np.array(I.shape[:-1])/100
    Figure, Axis = plt.subplots(1,1,figsize=FigSize, dpi=100)
    Axis.imshow(np.round(Categories[:,:,-3:] * 255).astype('int'))
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(MaskName, dpi=100)
    plt.close(Figure)
    
    Seg = np.zeros(S.shape + (4,))
    Seg[:,:,:-1] = Categories[:,:,2:] * 255
    BG = Categories[:,:,1].astype('bool')
    Seg[:,:,-1][~BG] = 255
    Seg[:,:,1][Seg[:,:,2] == 255] = 255

    FigSize = np.array(I.shape[:-1])/100
    Figure, Axis = plt.subplots(1,1,figsize=FigSize, dpi=100)
    Axis.imshow(I)
    Axis.imshow(Seg.astype('uint8'), alpha=0.5)
    Axis.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(FigName, dpi=100)
    plt.close(Figure)

    return

#%% Main
# Main part

def Main():

    # Set paths
    MainDir = Path(__file__).parent / '..' / '03_Results'
    DataDir = MainDir / 'ROIs_Selection'
    ResDir = MainDir / 'Segmentation'
    Path.mkdir(ResDir, exist_ok=True)

    # Load classifiers
    RFc = joblib.load(MainDir / 'RandomForest.joblib')
    RFc.verbose = 0
    Unet = load_model(MainDir / 'UNet.hdf5')

    # List folders
    Folders = sorted([F for F in DataDir.iterdir() if Path.is_dir(F)])

    # Loop over every donor, every sample
    for Folder in Folders:
        Path.mkdir(ResDir / Folder.name, exist_ok=True)

        # List and loop over ROI images
        Files = sorted([F for F in Folder.iterdir() if '_' in F.name])
        for File in Files:

            # Read and segment image
            Image = io.imread(File)
            PredRF = Segmentation(Image, Unet, RFc)

            # Clean cement lines segmentation
            Pred = CleanSegmentation(PredRF)

            # Save segmentation results
            FigName = ResDir / Folder.name / File.name
            SaveSegmentation(Image, Pred, FigName)

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

