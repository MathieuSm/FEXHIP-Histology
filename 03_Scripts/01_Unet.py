#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform automatic segmentation using unet

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
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import load_model
from patchify import patchify, unpatchify
from skimage import io, morphology, measure
from Utils import Time, SetDirectories, Show
from keras import utils, layers, Model, callbacks
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
def CollectData(Directory):

    ROIs, Labels = [], []
    Folders = os.listdir(str(Directory))
    Folders.sort()
    Time.Process(1, 'Collect Data')
    for i, F in enumerate(Folders):
        Folder = Directory / F
        Images = [I for I in os.listdir(Folder) if I[:5] == 'Image']
        Images.sort()
        for I in Images:
            ROI = io.imread(str(Folder / I))
            Label = io.imread(str(Folder / ('Label' + I[5:])))
            ROIs.append(ROI)
            Labels.append(Label)
        Time.Update(i / len(Folders))

    ROIs = np.array(ROIs)
    Labels = np.expand_dims(Labels, -1)

    Labels[Labels==253] = 3
    Labels[Labels==254] = 2
    Labels[Labels==255] = 1
    
    Time.Process(0)

    return ROIs, Labels
def ConvulationBlock(Input, nFilters):

    Layer = layers.Conv2D(nFilters, 3, padding="same")(Input)
    Layer = layers.Activation("relu")(Layer)

    Layer = layers.Conv2D(nFilters, 3, padding="same")(Layer)
    Layer = layers.Activation("relu")(Layer)

    return Layer
def EncoderBlock(Input, nFilters):
    Layer = ConvulationBlock(Input, nFilters)
    Pool = layers.MaxPool2D((2, 2))(Layer)
    return Layer, Pool
def DecoderBlock(Input, SkipFeatures, nFilters):
    Layer = layers.Conv2DTranspose(nFilters, (2, 2), strides=2, padding="same")(Input)
    Layer = layers.Concatenate()([Layer, SkipFeatures])
    Layer = ConvulationBlock(Layer, nFilters)
    return Layer
def BuildUNet(InputShape, nClasses, nFilters=[64, 128, 256, 512, 1024]):

    Input = layers.Input(InputShape)
    Block = []
    Block.append(EncoderBlock(Input, nFilters[0]))
    for i, nFilter in enumerate(nFilters[1:-1]):
        Block.append(EncoderBlock(Block[i][1], nFilter))

    Bridge = ConvulationBlock(Block[-1][1], nFilters[-1])
    D = DecoderBlock(Bridge, Block[-1][0], nFilters[-2])

    for i, nFilter in enumerate(nFilters[-3::-1]):
        D = DecoderBlock(D, Block[-i+2][0], nFilter)

    if nClasses == 2:  #Binary
      Activation = 'sigmoid'
    else:
      Activation = 'softmax'

    Outputs = layers.Conv2D(nClasses, 1, padding="same", activation=Activation)(D)
    UNet = Model(Input, Outputs, name='U-Net')
    return UNet
def PlotHistory(History, ResultsDir):

    Loss = History.history['loss']
    ValLoss = History.history['val_loss']
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(range(1, len(Loss) + 1), Loss, color=(0, 0, 1), marker='o', linestyle='--', label='Training loss')
    Axis.plot(range(1, len(Loss) + 1), ValLoss, color=(1, 0, 0), marker='o', linestyle='--', label='Validation loss')
    Axis.set_xlabel('Epochs')
    Axis.set_ylabel('Loss')
    Axis.legend()
    plt.savefig(str(ResultsDir / 'Loss.png'))
    plt.show()

    Accuracy = History.history['accuracy']
    ValAccuracy = History.history['val_accuracy']
    Figure, Axis = plt.subplots(1, 1)
    Axis.plot(range(1, len(Accuracy) + 1), Accuracy, color=(0, 0, 1), marker='o', linestyle='--', label='Training accuracy')
    Axis.plot(range(1, len(Accuracy) + 1), ValAccuracy, color=(1, 0, 0), marker='o', linestyle='--', label='Validation accuracy')
    Axis.set_xlabel('Epochs')
    Axis.set_ylabel('Accuracy')
    Axis.legend()
    plt.savefig(str(ResultsDir / 'Accuracy.png'))
    plt.show()

    return
def PlotConfusionMatrix(GroundTruth, Results, Ticks):

    CM = metrics.confusion_matrix(GroundTruth, Results, normalize=None)
    CM2 = metrics.confusion_matrix(GroundTruth, Results, normalize='true')
    CM3 = metrics.confusion_matrix(GroundTruth, Results, normalize='pred')
    VSpace = 0.2

    Figure, Axis = plt.subplots(1, 1, figsize=(5.5, 4.5))
    Axis.matshow(CM3, cmap='binary', alpha=0.33)
    for Row in range(CM.shape[0]):
        for Column in range(CM.shape[1]):
            Axis.text(x=Row, y=Column, position=(Row, Column), va='center', ha='center', s=CM[Row, Column])
            Axis.text(x=Row, y=Column, position=(Row, Column + VSpace), va='center', ha='center',
                      s=round(CM2[Row, Column], 2), color=(0, 0, 1))
            Axis.text(x=Row, y=Column, position=(Row, Column - VSpace), va='center', ha='center',
                      s=round(CM3[Row, Column], 2), color=(1, 0, 0))
    Axis.xaxis.set_ticks_position('bottom')
    Axis.set_xticks(np.arange(len(Ticks)), Ticks)
    Axis.set_yticks(np.arange(len(Ticks)), Ticks)
    Axis.set_ylim([-0.49, CM.shape[0] - 0.5])
    Axis.set_title('Total: ' + str(GroundTruth[GroundTruth > 0].size))
    Axis.set_xlabel('Ground Truth', color=(0, 0, 1))
    Axis.set_ylabel('Predictions', color=(1, 0, 0))
    plt.show()

    return CM
def SegmentationData(Masks):

    Osteocytes = pd.DataFrame(columns=['Number','Mean Area'])
    HCanals = pd.DataFrame(columns=['Number','Mean Area'])
    CMLines = pd.DataFrame(columns=['Number','Total Area'])
    LineThickness = morphology.disk(2)
    Time.Process(1, 'Get seg. data')
    for i, Mask in enumerate(Masks):

        Time.Update((i+1)/len(Masks))

        Labels = morphology.label(Mask == 1)
        Props = measure.regionprops_table(Labels, properties=['area'])
        Props = pd.DataFrame(Props)
        Osteocytes.loc[i] = [len(Props),Props.mean().values[0]]

        Labels = morphology.label(Mask == 2)
        Props = measure.regionprops_table(Labels, properties=['area'])
        Props = pd.DataFrame(Props)
        HCanals.loc[i] = [len(Props),Props.mean().values[0]]

        Erode = morphology.binary_erosion(Mask == 3, morphology.disk(1))
        Skeleton = morphology.skeletonize(Erode)
        CMLine = morphology.binary_dilation(Skeleton, LineThickness)
        Labels = morphology.label(CMLine)
        Props = measure.regionprops_table(Labels, properties=['area'])
        Props = pd.DataFrame(Props)
        CMLines.loc[i] = [len(Props),Props.sum().values[0]]

    Time.Process(0)

    return Osteocytes, HCanals, CMLines
def SegmentPrediction(Thresholds, Predictions, Data):

    Osteocytes, HCanals, CMLines = Data

    OsteocytesP = pd.DataFrame(columns=['Number','Mean Area'])
    HCanalsP = pd.DataFrame(columns=['Number','Mean Area'])
    CMLinesP = pd.DataFrame(columns=['Number','Total Area'])
    Totals = np.zeros(Thresholds.shape[0])
    for iT, Threshold in enumerate(Thresholds):
        for iP, Prediction in enumerate(Predictions):
            Segmented = np.copy(Prediction)
            Segment = np.zeros(4, int)
            for i, P in enumerate([OsteocytesP, HCanalsP, CMLinesP]):
                Segment[1+i] = 1
                Mask = Segmented[:,:,1+i] >= Threshold[i]
                Labels = morphology.label(Mask)
                Props = measure.regionprops_table(Labels, properties=['area'])
                Props = pd.DataFrame(Props)
                if np.isnan(Props.mean().values[0]):
                    P.loc[iP] = [0,0]
                elif i == 2:
                    P.loc[iP] = [len(Props),Props.sum().values[0]]
                else:
                    P.loc[iP] = [len(Props),Props.mean().values[0]]

        # Test 1 minimze total error
        ErrorsO = np.abs(OsteocytesP['Number'] / Osteocytes['Number'] - 1)
        ErrorsH = np.abs(HCanalsP['Number'] / HCanals['Number'] - 1)
        ErrorsC = np.abs(CMLinesP['Total Area'] / CMLines['Total Area'] - 1)
        Total = sum([ErrorsO.sum(), ErrorsH.sum(), ErrorsC.sum()])
        Totals[iT] = Total

    return Totals

#%% Main
# Main part

def Main():

    # Set directory and data
    WD, DD, SD, RD = SetDirectories('FEXHIP-Histology')
    ResultsDir = RD / '03_Segmentation'
    os.makedirs(ResultsDir, exist_ok=True)
    Directory = DD / '03_ManualSegmentation'
    Masks, Images, Data = CollectMasks(Directory)

    # Get preprocessed data
    ROIs, Labels = CollectData(RD / '02_Preprocessing' / 'Augmentation')
    Size = Labels.shape[-2]

    # Define class weights
    Counts = np.unique(Labels, return_counts=True)[1]
    cWeights = 1 / (Counts / sum(Counts) * len(Counts))
    # cWeights = {V:round(W,2) for V, W in zip(Values, Weights)}

    # Split into train and test data
    LabelsCat = utils.to_categorical(Labels)
    XTrain, XTest, YTrain, YTest = train_test_split(ROIs/255, LabelsCat, random_state=42)

    # Set sample weights according to manual segmentation
    AugmentationFactor = 16
    Operators = np.repeat(Data['Operator'].values, AugmentationFactor)
    Scores = pd.read_csv(str(RD / '02_Scores.csv'), index_col=[0])
    sWeights = np.zeros(Labels.shape)
    for i, Op in enumerate(Operators):
        if Op in Scores.index:
            Score = Scores[Scores.index == Op].values[0]
        else:
            Score = Scores.mean().values
        Mask = Labels[i,:,:,0]
        BG = np.where(Mask == 0)
        OS = np.where(Mask == 1)
        HC = np.where(Mask == 2)
        CL = np.where(Mask == 3)
        sWeights[i][BG] = Score.mean()
        sWeights[i][OS] = Score[0]
        sWeights[i][HC] = Score[1]
        sWeights[i][CL] = Score[2]
    WeightsTrain = train_test_split(sWeights, random_state=42)[0]

    # Build UNet
    UNet = BuildUNet(XTrain.shape[1:], YTrain.shape[-1])
    UNet.compile(optimizer='adam',
                 loss='binary_focal_crossentropy',
                 metrics=['accuracy'],
                 loss_weights=cWeights)
    print(UNet.summary())
    EarlyStop = callbacks.EarlyStopping(monitor='accuracy', patience=50)
    ModelName = str(ResultsDir / 'UNet.hdf5')
    CheckPoint = callbacks.ModelCheckpoint(ModelName, monitor='accuracy', mode='max', save_best_only=True)
    History = UNet.fit(XTrain,YTrain, validation_data=(XTest,YTest),
                       verbose=2, epochs=350, workers=4,
                       batch_size=8, steps_per_epoch=20,
                       callbacks=[EarlyStop, CheckPoint],
                       sample_weight=WeightsTrain)
    PlotHistory(History, ResultsDir)

    # Load best model
    UNet = load_model(str(ResultsDir / 'UNet.hdf5'))

    # Look at random testing image
    Random = np.random.randint(0, len(XTest)-1)
    TestImage = XTest[Random]
    TestLabel = YTest[Random]
    Prediction = UNet.predict(np.expand_dims(TestImage,0))[0]

    Figure, Axis = plt.subplots(1,3)
    Axis[0].imshow(TestImage)
    Axis[1].imshow(TestLabel[:,:,1:])
    Axis[2].imshow(Prediction[:,:,1:])
    for i in range(3):
        Axis[i].axis('off')
    plt.tight_layout()
    plt.show()

    # # Vizualize filters
    # Weights = []
    # for L in UNet.layers:
    #     if 'transpose' in L.name:
    #         pass
    #     elif 'conv2d_' in L.name:
    #         if np.mod(int(L.name[7:]), 2):
    #             Weights.append(L.get_weights()[0])

    # for W in Weights:

    #     W = np.transpose(W)
    #     W = (W - W.min()) / (W.max() - W.min())

    #     Figure, Axis = plt.subplots(3)
    #     for i in range(3):
    #         Axis[i].imshow(W[:,:,i])
    #         Axis[i].set_xticks([])
    #         Axis[i].set_yticks([])
    #     plt.show()

    # Collect UNet predictions and save
    PredDir = ResultsDir / 'Unet'
    os.makedirs(PredDir, exist_ok=True)
    for i, I in enumerate(Images):

        # Compute patch size
        NPatches = np.ceil(np.array(I.shape)[:2] / np.array((Size, Size))).astype(int)
        Overlap = NPatches * np.array((Size, Size)) - np.array(I.shape)[:2]
        Step = np.append(Size - Overlap, [3])
    
        # Separate image into patches to fit UNet
        Patches = patchify(I, (Size, Size, 3), step=Step)
        Seg = np.zeros(np.concatenate([I.shape[:-1], [4]]), float)
        for Xi, Px in enumerate(Patches):
            for Yi, Py in enumerate(Px):
                Pred = UNet.predict(Py / 255)[0]
                X1 = Xi*Step[0]
                X2 = Size + Xi*Step[0]
                Y1 = Yi*Step[1]
                Y2 = Size + Yi*Step[1]
                Seg[X1:X2, Y1:Y2] += Pred

        Max = np.argmax(Seg, axis=-1)
        Vals = np.unique(Max)
        Prediction = np.zeros(Seg.shape, 'uint8')
        for V in Vals:
            Mask = np.where(Max == V)
            Prediction[Mask[0],Mask[1],V] = 255

        IName = str(PredDir / str('Seg_' + '%02d' % (i) + '.png'))
        io.imsave(IName, Prediction[:,:,1:].astype('uint8'))


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
