#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script runs the interactive registration of sample
    and draw the cutting lines according to a given angle

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: May 2023
    """

#%% Imports
# Modules import

import os
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from Utils import Time, SetDirectories

from pylatex import Document, Figure, SubFigure, NoEscape, Command
from pylatex.package import Package

#%% Functions
# Define functions

def FillHoles(Image, Radius=5):

    Dimension = Image.GetDimension()

    if Dimension == 3:

        Crop = sitk.Crop(Image, (0, 0, Radius), (0, 0, Radius))
        Padded = sitk.ConstantPad(Crop, (0, 0, Radius), (0, 0, Radius), 2)
        Padded = sitk.ConstantPad(Padded, (Radius, Radius, Radius), (Radius, Radius, Radius))

        DilateFilter = sitk.BinaryDilateImageFilter()
        DilateFilter.SetForegroundValue(2)
        DilateFilter.SetKernelRadius(Radius)
        Dilated = DilateFilter.Execute(Padded)

        FillFilter = sitk.BinaryFillholeImageFilter()
        FillFilter.SetForegroundValue(2)
        Filled = FillFilter.Execute(Dilated)

        ErodeFilter = sitk.BinaryErodeImageFilter()
        ErodeFilter.SetForegroundValue(2)
        ErodeFilter.SetKernelRadius(Radius)
        Eroded = ErodeFilter.Execute(Filled)

        BinImage = sitk.Crop(Eroded, (Radius, Radius, Radius), (Radius, Radius, Radius))

    elif Dimension == 2:

        Padded = sitk.ConstantPad(Image, (Radius, Radius), (Radius, Radius))

        DilateFilter = sitk.BinaryDilateImageFilter()
        DilateFilter.SetForegroundValue(2)
        DilateFilter.SetKernelRadius(Radius)
        Dilated = DilateFilter.Execute(Padded)

        FillFilter = sitk.BinaryFillholeImageFilter()
        FillFilter.SetForegroundValue(2)
        Filled = FillFilter.Execute(Dilated)

        ErodeFilter = sitk.BinaryErodeImageFilter()
        ErodeFilter.SetForegroundValue(2)
        ErodeFilter.SetKernelRadius(Radius)
        Eroded = ErodeFilter.Execute(Dilated)

        BinImage = sitk.Crop(Eroded, (Radius, Radius), (Radius, Radius))

    return BinImage

def PrintImage(Image, FileName, pCOG, X_Line, Y_Line, Show=True):
    ImageSize = np.array(Image.GetSize())
    ImageSpacing = np.array(Image.GetSpacing())
    PhysicalSize = ImageSize * ImageSpacing

    Inch = 25.4
    mm = 1 / Inch
    Subsampling = 1
    DPI = np.round(1 / ImageSpacing / Subsampling * Inch)[0]
    Margins = np.array([0., 0.])
    FigureSize = ImageSpacing * ImageSize * mm

    RealMargins = [Margins[0] / 2 * PhysicalSize[0] / (FigureSize[0] * Inch),
                    Margins[1] / 2 * PhysicalSize[1] / (FigureSize[1] * Inch),
                    1 - Margins[0] / 2 * PhysicalSize[0] / (FigureSize[0] * Inch),
                    1 - Margins[1] / 2 * PhysicalSize[1] / (FigureSize[1] * Inch)]

    Figure, Axis = plt.subplots(1,1, figsize=FigureSize, dpi=int(DPI))
    Axis.imshow(sitk.GetArrayFromImage(Image), cmap='bone')
    Axis.plot(pCOG[0] + X_Line, pCOG[1] + Y_Line, color=(1, 0, 0))
    Axis.plot(pCOG[0] - X_Line, pCOG[1] + Y_Line, color=(1, 0, 0))
    Axis.plot(pCOG[0] + X_Line, pCOG[1] - Y_Line, color=(1, 0, 0))
    Axis.plot(pCOG[0] - X_Line, pCOG[1] - Y_Line, color=(1, 0, 0))
    Axis.set_xlim([0,Image.GetSize()[0]])
    Axis.set_ylim([Image.GetSize()[1],0])
    Axis.axis('off')
    plt.subplots_adjust(RealMargins[0], RealMargins[1], RealMargins[2], RealMargins[3])
    plt.savefig(FileName, dpi=int(DPI))
    if Show:
        plt.show()
    else:
        plt.close(Figure)

#%% Main
# Main part

def Main():

    # Set directory and load sample list
    WD, DD, SD, RD = SetDirectories('FEXHIP-Histology')
    SampleList = pd.read_csv(str(DD / 'SampleList.csv'))
    uCTD = DD / '01_uCTScans'
    OutD = RD / '01_Quadrants'
    Angle = 60

    Exceptions = ['389L','397L','427L']

    Time.Process(1, 'Print lines')
    for Index, Row in SampleList.iterrows():

        Sample = str(Row['Sample ID'])
        uCT = Row['uCT'] + '.mhd'
        Text = 'Sample ' + Sample + ' ' + Row['Side']
        Time.Update(Index / len(SampleList), Text)

        # Read corresponding sample uct
        Scan = sitk.ReadImage(str(uCTD / uCT))
        Array = sitk.GetArrayFromImage(Scan)

        # Binarize image to find area centroid
        OtsuFilter = sitk.OtsuMultipleThresholdsImageFilter()
        OtsuFilter.SetNumberOfThresholds(2)
        BinImage = OtsuFilter.Execute(Scan)

        # Perform binary dilation and erosion to find center for proximal and distal slices
        Filled = FillHoles(BinImage)

        # Use distal slice for orientation computation
        Slice = int(round(Filled.GetSize()[2]*0.1))

        if Sample + Row['Side'] in Exceptions:
            Slice = -Slice

        # Get orientation of filled slice
        LabStat = sitk.LabelIntensityStatisticsImageFilter()
        LabStat.Execute(Filled[:,:,Slice], Filled[:,:,Slice])
        COGFilled = LabStat.GetCenterOfGravity(2)
        X1, Y1, X2, Y2 = LabStat.GetPrincipalAxes(2)
        OrientationAngle = -np.arctan(-X1 / Y1)

        # Create according rotation transform
        R = np.array([[np.cos(OrientationAngle), -np.sin(OrientationAngle), 0],
                      [np.sin(OrientationAngle),  np.cos(OrientationAngle), 0],
                      [                       0,                         0, 1]])

        Transform = sitk.AffineTransform(3)
        Transform.SetMatrix(R.ravel())
        Transform.SetCenter((COGFilled[0], COGFilled[1], 0))
        Transform.SetTranslation((0, 0, 0))

        # Resample image (and pad to avoid out-of-view bone parts)
        Resampler = sitk.ResampleImageFilter()
        Pad = 50
        Resampler.SetReferenceImage(sitk.ConstantPad(Scan,(Pad,Pad,0),(Pad,Pad,0)))
        Resampler.SetTransform(Transform.GetInverse())
        Rotated = Resampler.Execute(Scan)

        # Check if inferior if on left
        BinImage = OtsuFilter.Execute(Rotated)
        LabStat.Execute(BinImage[:,:,Slice], BinImage[:,:,Slice])
        COGBin = LabStat.GetCenterOfGravity(2)

        if COGBin[0] > COGFilled[0]:
            Transform.SetMatrix([-1, 0, 0, 0, -1, 0, 0, 0, 1])
            Resampler.SetTransform(Transform.GetInverse())
            Rotated = Resampler.Execute(Rotated)

        # Filter and crop imag to bone
        Gauss = sitk.DiscreteGaussianImageFilter()
        Gauss.SetVariance(0.05)
        Gauss.SetMaximumKernelWidth(10)
        Smooth = Gauss.Execute(Rotated[:,:,Slice])

        BinImage = OtsuFilter.Execute(Smooth)
        Array = sitk.GetArrayFromImage(BinImage)
        Coords = np.where(Array == 2)
        Cropped = BinImage[Coords[1].min()-5: Coords[1].max()+5,
                         Coords[0].min()-5: Coords[0].max()+5]

        # Get center of gravity (COG) of filled slice
        COG = Filled[:,:,Slice].TransformPhysicalPointToIndex(COGFilled)
        COG = (COG[0] - Coords[1].min() + Pad,
               COG[1] - Coords[0].min() + Pad)


        # Compute cutting lines
        X_Line = np.linspace(0,Cropped.GetSize()[0],100)
        Y_Line = X_Line * np.tan(Angle/2 * np.pi/180)

        # Print image with correct size
        FileName = str(OutD / (Sample + '_' + Row['Side'] + '.png'))
        PrintImage(1 - (Cropped==2), FileName, COG, X_Line, Y_Line, Show=False)

        SampleList.loc[Index, 'Width'] = Cropped.GetSize()[0] * Cropped.GetSpacing()[0]

    Time.Process(0, 'Lines printed!')

    # Write pdf file
    Time.Process(1, 'Write PDF')
    Files = [F for F in os.listdir(OutD) if F.endswith('.png')]
    Files.sort()
    Factor = 1.2

    Doc = Document(default_filepath=str(RD / '01_Quadrants'))
    for Index, File in enumerate(Files):

        Time.Update(Index / len(Files))

        Image = str(OutD / File)

        if np.mod(Index, 2) == 0:
            Images = [Image]
            SubCaptions = [File[:-4]]
            Width = [str(round(SampleList.loc[Index,'Width'] * Factor,1))]
        
        else:
            Images.append(Image)
            SubCaptions.append(File[:-4])
            Width.append(str(round(SampleList.loc[Index,'Width'] * Factor,1)))


            with Doc.create(Figure(position='h!')):
                Doc.append(Command('centering'))
                for i, Image in enumerate(Images):
                    SubFig = SubFigure(position='b',width=NoEscape(Width[i] + ' mm'))
                    with Doc.create(SubFig) as SF:
                        SF.add_image(Image)
                        SF.add_caption(SubCaptions[i])
                    
                    if i == 0:
                        Doc.append(Command('hfill'))

    Doc.packages.append(Package('subcaption', options='aboveskip=0pt, labelformat=empty'))
            
    Doc.generate_pdf(clean_tex=False)

    Time.Process(0)

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
