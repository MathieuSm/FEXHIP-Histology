#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script aims to test the module instalation

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
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.morphology import disk
from sklearn.model_selection import KFold
from keras import layers, Model, callbacks
from scipy.ndimage import maximum_filter as mf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skimage import io, feature, color, filters

from Utils import Time, Training

#%% Functions
# Define functions

def ConvolutionBlock(Input:layers.Layer, nFilters:int):

    """
    Classical convolutional block used in simple U-net models
    """

    Layer = layers.Conv2D(nFilters, 3, padding="same")(Input)
    Layer = layers.Activation("relu")(Layer)

    Layer = layers.Conv2D(nFilters, 3, padding="same")(Layer)
    Layer = layers.Activation("relu")(Layer)

    return Layer
def EncoderBlock(Input:layers.Layer, nFilters:int):

    """
    Classical encoder block used in simple U-net models
    """

    Layer = ConvolutionBlock(Input, nFilters)
    Pool = layers.MaxPool2D((2, 2))(Layer)
    return Layer, Pool
def DecoderBlock(Input:layers.Layer, SkipFeatures:layers.Layer, nFilters:int):
    
    """
    Classical decoder block used in simple U-net models    
    """

    Layer = layers.Conv2DTranspose(nFilters, (2, 2), strides=2, padding="same")(Input)
    Layer = layers.Concatenate()([Layer, SkipFeatures])
    Layer = ConvolutionBlock(Layer, nFilters)
    return Layer
def BuildUnet(InputShape:tuple, nClasses:int, nFilters=[64, 128, 256, 512, 1024]) -> Model:

    """
    Builds simple U-net model for semantic segmentataion
    Model doesn't comprise dropout or batch normalization to keep
    architecture as simple as possible
    :param InputShape: tuple of 2 numbers defining the U-net input shape
    :param nClasses: integer defining the number of classes to segment
    :param nFilters: list of number of filters for each layer
    :return Unet: keras unet model
    """
    
    Input = layers.Input(InputShape)
    Block = []
    Block.append(EncoderBlock(Input, nFilters[0]))
    for i, nFilter in enumerate(nFilters[1:-1]):
        Block.append(EncoderBlock(Block[i][1], nFilter))

    Bridge = ConvolutionBlock(Block[-1][1], nFilters[-1])
    D = DecoderBlock(Bridge, Block[-1][0], nFilters[-2])

    for i, nFilter in enumerate(nFilters[-3::-1]):
        D = DecoderBlock(D, Block[-i+2][0], nFilter)

    # If binary classification, uses sigmoid activation function
    if nClasses == 2:
      Activation = 'sigmoid'
    else:
      Activation = 'softmax'

    Outputs = layers.Conv2D(nClasses, 1, padding="same", activation=Activation)(D)
    Unet = Model(Input, Outputs, name='U-Net')
    return Unet

#%% Main
# Main part

def Main():

    # List training data
    ResultsDir = Path(__file__).parent / '..' / '03_Results'
    DataPath = ResultsDir / 'Data_Augmentation'
    Images, Labels = Training.GetData(DataPath)

    # Get sample weights
    sWeights, YData = Training.SetWeights(Labels)

    # Define class weights
    Counts = np.sum(YData, axis=(0,1,2))
    cWeights = 1 / (Counts / sum(Counts) * len(Counts))

    # Split into train and test data
    XTrain, XTest, YTrain, YTest = train_test_split(Images/255, YData, random_state=42)
    WTrain, WTest = train_test_split(sWeights, random_state=42)

    # # Build UNet
    # Unet = BuildUnet(XTrain.shape[1:], YTrain.shape[-1])
    # Unet.compile(optimizer='adam',
    #              loss='binary_focal_crossentropy',
    #              metrics=['accuracy'],
    #              loss_weights=cWeights,
    #              weighted_metrics=['accuracy'])
    # print(Unet.summary())
    # EarlyStop = callbacks.EarlyStopping(monitor='accuracy', patience=100)
    # ModelName = str(ResultsDir / 'UNet_x.hdf5')
    # CheckPoint = callbacks.ModelCheckpoint(ModelName, monitor='accuracy',
    #                                        mode='max', save_best_only=True)
    # History = Unet.fit(XTrain,YTrain, validation_data=(XTest,YTest,WTest),
    #                    verbose=2, epochs=500, workers=8,
    #                    batch_size=8, steps_per_epoch=20,
    #                    callbacks=[EarlyStop, CheckPoint],
    #                    sample_weight=WTrain)
    # HistoryDf = pd.DataFrame(History.history)
    # HistoryDf.to_csv(ResultsDir / 'History_x.csv', index=False)

    # Load best model
    Unet = load_model(ResultsDir / 'UNet.hdf5')

    # # Look at random testing image
    # Random = np.random.randint(0, len(XTest)-1)
    # TestImage = XTest[Random]
    # TestLabel = YTest[Random] * 255
    # Prediction = Unet.predict(np.expand_dims(TestImage,0))[0]
    # ArgMax = np.argmax(Prediction, axis=-1)

    # Figure, Axis = plt.subplots(1,3)
    # Axis[0].imshow(TestImage)
    # Axis[1].imshow(TestLabel[:,:,1:])
    # Axis[2].imshow(Prediction[:,:,1:])
    # # Axis[2].imshow(ArgMax)
    # for i in range(3):
    #     Axis[i].axis('off')
    # plt.tight_layout()
    # plt.show()

    # from skimage.filters import threshold_otsu
    # Threshold1 = threshold_otsu(Prediction[:,:,-1])
    # Cl = Prediction[:,:,-1] > Threshold1
    # Im = np.zeros(Cl.shape + (4,))
    # Im[Cl] = [1,0,0,1]

    # %matplotlib widget
    # Figure, Axis = plt.subplots(1,1)
    # Axis.imshow(TestImage)
    # ClAxis = Axis.imshow(Im)
    # Axis.axis('off')

    # Sl1ax = Figure.add_axes([0.2, 0.0, 0.6, 0.02])

    # from matplotlib.widgets import Slider
    # Slider1 = Slider(ax=Sl1ax, label='Cement Lines', valmin=0,
    #                  valmax=1,
    #                  valinit=Threshold1,
    #                  orientation='horizontal')

    # def Update1(val):
    #     Cl = Prediction[:,:,-1] > Slider1.val
    #     Im = np.zeros(Cl.shape + (4,))
    #     Im[Cl] = [1,0,0,1]
    #     ClAxis.set_data(Im)
    #     Figure.canvas.flush_events()

    # Slider1.on_changed(Update1)

    # plt.show(Figure)

    # Collect classes probability
    Prob = []
    Time.Process(1,'UNet estimation')
    for i, ROI in enumerate(Images):
        ROI = np.expand_dims(ROI, 0)
        Prob.append(Unet.predict(ROI / 255, verbose=0)[0])
        Time.Update(i / len(Images))

    Prob = np.array(Prob)
    Time.Process(0)

    # Extract features for random forest fit
    Features = []
    Kwargs = {'channel_axis':-1,
              'sigma_min':2.0,
              'num_sigma':3}
    Time.Process(1,'Extract features')
    for i, ROI in enumerate(Images):
        Feature = feature.multiscale_basic_features(ROI,**Kwargs)
        Features.append(Feature)
        Time.Update(i / len(Images))

    Features = np.array(Features)
    Time.Process(0)

    # Balance and subsample data for faster fit
    aLabels = np.argmax(YData, axis=-1)
    Values, Counts = np.unique(aLabels, return_counts=True)
    Indices = pd.DataFrame(aLabels.ravel()).groupby(0).sample(min(Counts)).index
    
    # Reshape array and subsample
    Prob = Prob.reshape(-1, Prob.shape[-1])
    Features = Features.reshape(-1, Features.shape[-1])
    # SubFeatures = np.concatenate([Features[Indices], Prob[Indices]], axis=1)
    SubFeatures = Features[Indices]
    SubLabels = aLabels.ravel()[Indices]
    SubWeights = sWeights.ravel()[Indices]

    # Instanciate random forest classifier
    RFc = RandomForestClassifier(n_estimators=50,
                                 oob_score=True,
                                 max_depth=10,
                                 n_jobs=-1,
                                 verbose=0,
                                 class_weight='balanced')

    # Fit random forest classifier with all data and save it
    RFc.verbose = 2
    XTrain = SubFeatures
    WTrain = SubWeights.ravel()
    YTrain = SubLabels.ravel()
    WTrain = WTrain * cWeights[YTrain]      # Multiply by class weights
    RFc.fit(XTrain, YTrain+1, sample_weight=WTrain)
    joblib.dump(RFc, str(ResultsDir / 'RandomForest_Px.joblib'))

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
