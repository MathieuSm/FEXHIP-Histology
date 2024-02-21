#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script performs analysis of segmentation results

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

import argparse
import numpy as np
import pandas as pd
from Utils import Time
from pathlib import Path
from skimage import io, measure, morphology

#%% Main
# Main part

def Main():

    # Record elapsed time
    Time.Process(1, 'Collect Seg Data')

    # Set paths
    MainDir = Path(__file__).parent / '..' / '03_Results'
    DataDir = MainDir / 'Segmentation'

    # List folders and create data frame
    Folders = sorted([F for F in DataDir.iterdir() if Path.is_dir(F)])
    Products = [[F.name for F in Folders],['Right','Left'], np.arange(1,13)]
    Indices = pd.MultiIndex.from_product(Products)
    Tissues = ['Haversian canals', 'Osteocytes', 'Cement lines']
    Variables = ['Density (%)', 'Area (um2)', 'Number (-)', 'Thickness (um)']
    Columns = pd.MultiIndex.from_product([Tissues, Variables])
    Data = pd.DataFrame(index=Indices, columns=Columns)

    # Pixel spacing
    PS = 1.046

    # Loop over every donor, every sample
    for iF, Folder in enumerate(Folders):

        # List and loop over ROI images
        Files = sorted([F for F in Folder.iterdir() if '_Mask' in F.name])
        for File in Files:

            # Set index
            Index = (Folder.name,
                     File.name.split('_')[0],
                     int(File.name.split('_')[1]) + 1)

            # Read and segment image
            Mask = io.imread(File)

            # Get individual masks
            OS = Mask[:,:,0] == 255
            HC = Mask[:,:,1] == 255
            CL = Mask[:,:,2] == 255

            # Measure osteocytes area
            Labels = measure.label(OS)
            RP = measure.regionprops_table(Labels, properties=['area'])
            RP = pd.DataFrame(RP)
            Data.loc[Index]['Osteocytes','Area (um2)'] = RP['area'].mean()*PS**2
            Data.loc[Index]['Osteocytes','Number (-)'] = len(RP)

            # Measure Haversian canals not connected to border
            Labels = measure.label(HC)
            RP1 = measure.regionprops_table(Labels, properties=['area', 'label'])
            RP1 = pd.DataFrame(RP1)
            RP2 = measure.regionprops_table(Labels[1:-1,1:-1], properties=['area','label'])
            RP2 = pd.DataFrame(RP2)

            # Keep Haversian canals not connected to border
            RP = pd.DataFrame(index=RP2['label'], columns=['area'])
            for l in RP.index:
                A1 = RP1[RP1['label'] == l]['area'].values[0]
                A2 = RP2[RP2['label'] == l]['area'].values[0]
                if A1 == A2:
                    RP.loc[l] = A1
            RP = RP.dropna()

            Data.loc[Index]['Haversian canals','Area (um2)'] = RP['area'].mean()*PS**2
            Data.loc[Index]['Haversian canals','Number (-)'] = len(RP)

            # Measure cement lines thickness
            MA, D = morphology.medial_axis(CL, return_distance=True)
            Data.loc[Index]['Cement lines','Thickness (um)'] = np.mean(D[MA])*PS

            # Compute relative densities
            HC = HC.sum() / HC.size * 100
            OS = OS.sum() / OS.size * 100
            CL = CL.sum() / CL.size * 100

            Data.loc[Index]['Haversian canals', 'Density (%)'] = HC
            Data.loc[Index]['Osteocytes', 'Density (%)'] = OS / (1- HC/100)
            Data.loc[Index]['Cement lines', 'Density (%)'] = CL / (1 - HC/100)

        # Update time
        Time.Update((iF+1)/len(Folders))

    # Save data
    Data.to_csv(MainDir / 'SegmentationData.csv')

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

