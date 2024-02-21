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
from keras import utils
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage.morphology import disk
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from skimage import io, feature, morphology, measure

from Utils import Time, Unet_Probabilities, StainNA, Reference, CVAT

#%% Functions
# Define functions

def ShowKFolds(Results, Sites, FileName):

    NSplits = len(Results)

    for Metric in ['Precision', 'Recall']:
        
        Figure, Axis = plt.subplots(2,2, sharex=True, sharey=True, dpi=200)
        for i in range(2):
            for j in range(2):
                # Plot bars
                Axis[i,j].bar(np.arange(NSplits)+1,
                            Results[Metric][Sites[i*2+j]],
                            edgecolor=(0,0,1), color=(1, 1, 1, 0))

                # Plot mean value
                Mean = Results[Metric][Sites[i*2+j]].mean()
                Axis[i,j].plot(np.arange(NSplits)+1,
                            np.repeat(Mean, NSplits),
                            color=(1,0,0))
                
                # Plot standard deviation range
                Std = Results[Metric][Sites[i*2+j]].std()
                Axis[i,j].fill_between(np.arange(NSplits)+1,
                                    np.repeat(Mean - Std, NSplits),
                                    np.repeat(Mean + Std, NSplits),
                                    color=(0,0,0,0.2))
                
                # Set axis label and title
                Axis[i,j].set_title(Sites[i*2+j])

                Axis[1,j].set_xlabel('Folds')
                Axis[1,j].set_xticks(np.arange(NSplits)+1)
                Axis[i,0].set_ylabel(Metric)
        plt.savefig(FileName + '_' + Metric + '.png', dpi=200)
        plt.show()

    return
def FeaturesImportance(RFc):

    """
    Show random forest feature importances using
    impurity-based importance, also know as the
    Gini-importance.
    """

    # Analyse random forest features importances
    Sigmas = np.logspace(np.log2(2), np.log2(16), 3, base=2)
    Sigmas = ['Sigma ' + str(round(s)) for s in Sigmas]
    Channels = ['R', 'G', 'B']
    Features = ['I', 'E', 'H1', 'H2']
    Indices = pd.MultiIndex.from_product([Channels, Sigmas, Features])
    Importances = RFc.feature_importances_ * 100
    FI = pd.DataFrame(Importances[:-4], index=Indices, columns=['Importances'])
    # FI = pd.DataFrame(Importances,columns=['Importances'])

    # Individual contributions of unet
    Figure, Axis = plt.subplots(1,1, figsize=(4,4), dpi = 200)
    Axis.bar(0, Importances[-4], edgecolor=(1,0,0), facecolor=(0,0,0,0))
    Axis.bar(1, Importances[-3], edgecolor=(1,0,0), facecolor=(0,0,0,0))
    Axis.bar(2, Importances[-2], edgecolor=(1,0,0), facecolor=(0,0,0,0))
    Axis.bar(3, Importances[-1], edgecolor=(1,0,0), facecolor=(0,0,0,0))
    Axis.set_xticks(np.arange(4),[ 'Interstitial\nTissue', 'Osteocyte', 'Haversian\nCanal','Cement\nLine'])
    Axis.set_ylabel('Mean decrease in impurity (%)')
    plt.show(Figure)

    # Image basic multiscale individual feature contributions
    Figure, Axis = plt.subplots(1,3, sharey=True, figsize=(9,3), dpi=200)
    for i, (Idx, Df) in enumerate(FI.groupby(level=1)):
        # Axis[i].bar(np.arange(4), Df.loc['L'].values[:,0], edgecolor=(1,0,0), facecolor=(0,0,0,0))
        Axis[i].bar(np.arange(4), Df.loc['R'].values[:,0], edgecolor=(1,0,0), facecolor=(0,0,0,0))
        Axis[i].bar(np.arange(4), Df.loc['G'].values[:,0], edgecolor=(0,1,0), facecolor=(0,0,0,0))
        Axis[i].bar(np.arange(4), Df.loc['B'].values[:,0], edgecolor=(0,0,1), facecolor=(0,0,0,0))
        Axis[i].set_xticks(np.arange(4), Features)
        Axis[i].set_xlabel(Sigmas[i])
    Axis[1].plot([],color=(1,0,0), label='Red channel')
    Axis[1].plot([],color=(0,1,0), label='Green channel')
    Axis[1].plot([],color=(0,0,1), label='Blue channel')
    Axis[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncols=3)
    Axis[0].set_ylabel('Mean decrease in impurity (%)')
    plt.show(Figure)

    # Add Unet features importance
    FI.loc['U1'] = Importances[-4]
    FI.loc['U2'] = Importances[-3]
    FI.loc['U3'] = Importances[-2]
    FI.loc['U4'] = Importances[-1]
    FI.loc[''] = 100

    Sorted = FI.sort_values(by='Importances', ascending=False)
    CumSum = Sorted.cumsum()['Importances'] - 100

    Figure, Axis = plt.subplots(1,1, dpi=200)
    Axis.plot([4,4],[0,CumSum.values[4]], color=(0,0,0), linestyle='--')
    Axis.plot(np.arange(len(FI)), CumSum, color=(1,0,0))
    Axis.set_xticks([2,22], ['U-Net', 'Multi-scale basic features'])
    Axis.set_ylabel('Mean decrease in impurity (%)')
    plt.show(Figure)

    return FI
def ClassesDistribution(YData):

    Values, Counts = np.unique(YData, return_counts=True)

    Figure, Axis = plt.subplots(1,1, figsize=(4,4), dpi = 200)
    Axis.bar(Values, Counts, edgecolor=(1,0,0), facecolor=(0,0,0,0))
    Axis.set_xticks(Values,[ 'Interstitial\nTissue', 'Osteocyte', 'Haversian\nCanal','Cement\nLine'])
    Axis.set_ylabel('Number of pixels (-)')
    Axis.set_title(f'Total {sum(Counts)} pixels')
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.show(Figure)

    return
def GetNeighbours(Array):
    """
    Function used to get values of the neighbourhood pixels (based on numpy.roll)
    :param Array: numpy array
    :return: Neighbourhood pixels values
    """

    # Define a map for the neighbour index computation
    Map = np.array([[-1, 0], [ 1, 0],
                    [ 0,-1], [ 0, 1],
                    [-1,-1], [ 1, 1],
                    [-1, 1], [ 1,-1]])
    
    Neighbourhood = np.zeros(Array.shape + (len(Map),))
    PadArray = np.pad(Array, ((1, 1), (1, 1)))

    for i, Shift in enumerate(Map):
        Neighbourhood[:,:,i] = np.roll(PadArray, Shift, axis=(0,1))[1:-1,1:-1]

    return Neighbourhood, Map
def RemoveEndBranches(Skeleton):

    N = GetNeighbours(Skeleton)[0]
    EndPoints = N.sum(axis=-1) * Skeleton == 1
    while np.sum(EndPoints):
        Skeleton[EndPoints] = False
        N = GetNeighbours(Skeleton)[0]
        EndPoints = N.sum(axis=-1) * Skeleton == 1

    return Skeleton

#%% Main
# Main part

def Main():

    # Set directories
    TrainingDir = Path(__file__).parent / '..' / '01_Data' / 'Training'
    ModelDir = Path(__file__).parent / '..' / '03_Results'
    ResultsDir = ModelDir / 'Assessment'
    os.makedirs(ResultsDir, exist_ok=True)

    # List training data
    Names, Images, Labels = CVAT.GetData(TrainingDir)

    # Get common ROI
    CommonName, CommonROI, SegRatio, Indices = CVAT.CommonROI()

    # Collect individual segments
    OneHot = CVAT.OneHot(Labels)

    # Store operator score for each segment
    sWeights = CVAT.SampleWeights(OneHot, Indices, SegRatio)

    # Get data, one-hot encoded labels, sample and classes weights
    YData = np.argmax(OneHot, axis=-1) + 1

    # Keep only 1 occurence of common ROI
    ROIs = Images[~Indices]
    Masks = np.expand_dims(sWeights[~Indices],-1) * OneHot[~Indices]
    Truth = YData[~Indices]

    ROIs = np.concatenate([np.expand_dims(CommonROI, 0), ROIs])
    sWeights = np.concatenate([np.expand_dims(SegRatio, 0), Masks])
    YData = np.concatenate([np.expand_dims(YData[Indices][0], 0), Truth])

    # Define data
    Stain = [StainNA(R, Reference.Mean, Reference.Std) for R in ROIs]
    XData = np.array(Stain)

    # Load best model
    Unet = load_model(ModelDir / 'UNet.hdf5')

    # Collect classes probability
    Prob = []
    Time.Process(1,'UNet estimation')
    for i, ROI in enumerate(XData):
        Prediction = Unet_Probabilities(Unet, ROI)
        Prob.append(Prediction)
        Time.Update(i / len(XData))
    Prob = np.array(Prob)
    Time.Process(0)

    # Save U-Net predictions
    ImPred = np.argmax(Prob, axis=-1) + 1
    for i, P in enumerate(ImPred):

        S = P.shape
        Categories = utils.to_categorical(P)
        Seg = np.zeros((S[0],S[1],4))
        Seg[:,:,:-1] = Categories[:,:,2:] * 255
        BG = Categories[:,:,1].astype('bool')
        Seg[:,:,-1][~BG] = 255
        Seg[:,:,1][Seg[:,:,2] == 255] = 255

        IName = str(ResultsDir / str('Seg_' + '%02d' % (i) + '_UNet.png'))
        Figure, Axis = plt.subplots(1,1)
        Axis.imshow(ROIs[i])
        Axis.imshow(Seg.astype('uint8'), alpha=0.5)
        Axis.axis('off')
        plt.subplots_adjust(0,0,1,1)
        plt.savefig(IName)
        plt.close(Figure)

    # Extract features for random forest fit
    Features = []
    Kwargs = {'channel_axis':-1,
              'sigma_min':2.0,
              'num_sigma':3}
    Time.Process(1,'Extract features')
    for i, ROI in enumerate(XData):
        Feature = feature.multiscale_basic_features(ROI,**Kwargs)
        Features.append(Feature)
        Time.Update(i / len(XData))
    Features = np.array(Features)
    Time.Process(0)

    # Balance and subsample data for faster fit
    Values, Counts = np.unique(YData, return_counts=True)
    Indices = pd.DataFrame(YData.ravel()).groupby(0).sample(min(Counts//2)).index
    
    # Reshape array and subsample
    Prob = Prob.reshape(-1, Prob.shape[-1])
    Features = Features.reshape(-1, Features.shape[-1])
    SubFeatures = np.concatenate([Features[Indices], Prob[Indices]], axis=1)
    SubLabels = YData.ravel()[Indices]
    SubWeights = sWeights.ravel()[Indices]

    # Instanciate random forest classifier
    RFc = RandomForestClassifier(n_estimators=100,
                                oob_score=True,
                                n_jobs=-1,
                                verbose=0,
                                max_depth=20,
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
        WTrain = SubWeights[Train]
        XTest = SubFeatures[Test]
        YTest = SubLabels[Test]

        # Fit random forest
        RFc.fit(XTrain, YTrain, sample_weight=WTrain)

        # Perform prediction
        Pred = RFc.predict(XTest)
        
        # Compute scores
        RF_Precision = metrics.precision_score(Pred, YTest, average=None)
        RF_Recall = metrics.recall_score(Pred, YTest, average=None)
        Precision = [P for P in RF_Precision]
        Recall = [R for R in RF_Recall]
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
    ShowKFolds(FoldsScores, Sites, str(ResultsDir / 'KFolds'))

    # Load model fitted with all data and segment original images
    RFc = joblib.load(str(ModelDir / 'RandomForest.joblib'))
    Prediction = RFc.predict(np.concatenate([Features, Prob], axis=1))
    FI = FeaturesImportance(RFc)

    # Save random forest segmentation
    ImPred = Prediction.reshape(np.array(YData).shape)
    for i, P in enumerate(ImPred):

        S = P.shape
        Categories = utils.to_categorical(P)
        Seg = np.zeros((S[0],S[1],4))
        Seg[:,:,:-1] = Categories[:,:,2:] * 255
        BG = Categories[:,:,1].astype('bool')
        Seg[:,:,-1][~BG] = 255
        Seg[:,:,1][Seg[:,:,2] == 255] = 255

        IName = str(ResultsDir / str('Seg_' + '%02d' % (i) + '_RFo.png'))
        Figure, Axis = plt.subplots(1,1)
        Axis.imshow(ROIs[i])
        Axis.imshow(Seg.astype('uint8'), alpha=0.5)
        Axis.axis('off')
        plt.subplots_adjust(0,0,1,1)
        plt.savefig(IName)
        plt.close(Figure)

    # Perform morphological operations
    for i, P in enumerate(ImPred):

        Pc = P.copy()

        # Remove small cement line regions
        Cl = P == 4
        Os = P == 2
        Threshold = 1000
        Pad = np.pad(Cl+Os, ((1,1),(1,1)), mode='constant', constant_values=True)
        Regions = measure.label(Pad)
        Val, Co = np.unique(Regions, return_counts=True)
        Region = np.isin(Regions, Val[1:][Co[1:] > Threshold])
        Region = Region[1:-1,1:-1] * Cl
        Region = morphology.binary_erosion(Region, morphology.disk(1))
        P[(P == 4)*~Region] = 1
        P[(P == 4)*Region] = 4
        P[(P == 1)*~Region] = 1
        P[(P == 1)*Region] = 4

        # # Osteocytes
        # Os = P == 2
        # Threshold = 25
        # Erode = morphology.binary_erosion(Os, morphology.disk(3))
        # Dilate = morphology.binary_dilation(Erode, morphology.disk(2))
        # Regions = measure.label(Dilate)
        # Val, Co = np.unique(Regions, return_counts=True)
        # Region = np.isin(Regions, Val[1:][Co[1:] > Threshold])
        # Pc[(P == 2)*~Region] = 1
        # Pc[(P == 2)*Region] = 2
        # Pc[(P == 1)*~Region] = 1
        # Pc[(P == 1)*Region] = 2
    Prediction = ImPred.ravel()
    
    Precision = metrics.precision_score(Prediction, SubLabels, average=None)
    Recall = metrics.recall_score(Prediction, SubLabels, average=None)

    # Plot confusion matrix
    CMWeights = sWeights.reshape(-1, sWeights.shape[-1])
    CMWeights = np.max(CMWeights, axis=-1)
    CM = metrics.confusion_matrix(YData.ravel(),Prediction.ravel(),normalize=None)
    Recall = metrics.confusion_matrix(YData.ravel(),Prediction.ravel(),normalize='true')
    Precision = metrics.confusion_matrix(YData.ravel(),Prediction.ravel(),normalize='pred')
    VSpace = 0.2/2
    Ticks = ['Interstitial\nTissue', 'Osteocytes', 'Haversian\nCanals', 'Cement\nLine']
    Figure, Axis = plt.subplots(1,1, figsize=(5.5,4.5), dpi=200)
    # Axis.matshow(CM2, cmap='binary', alpha=0.33)
    for Row in range(CM.shape[0]):
        for Column in range(CM.shape[1]):
            # Axis.text(x=Row, y=Column, position=(Row,Column), va='center', ha='center', s=CM[Row, Column])
            Axis.text(x=Row, y=Column, position=(Row,Column+VSpace), va='center', ha='center', s=round(Recall[Row, Column],2), color=(0,0,1))
            Axis.text(x=Row, y=Column, position=(Row,Column-VSpace), va='center', ha='center', s=round(Precision[Row, Column],2), color=(1,0,0))
    Axis.xaxis.set_ticks_position('bottom')
    Axis.set_xticks(np.arange(len(Ticks)),Ticks)
    Axis.set_yticks(np.arange(len(Ticks)),Ticks)
    Axis.set_ylim([-0.49,CM.shape[0]-0.5])
    Axis.set_xlim([-0.49,CM.shape[0]-0.5])
    Axis.set_xlabel('Recall',color=(0,0,1))
    Axis.set_ylabel('Precision',color=(1,0,0))
    Axis.xaxis.set_label_position('top')
    Axis.yaxis.set_label_position('right')
    plt.tight_layout()
    plt.savefig(str(ResultsDir / 'ConfusionMatrix.png'), dpi=200)
    plt.show()

    # Save predictions
    for i, P in enumerate(ImPred):

        S = P.shape
        Categories = utils.to_categorical(P)
        Seg = np.zeros((S[0],S[1],4))
        Seg[:,:,:-1] = Categories[:,:,2:] * 255
        BG = Categories[:,:,1].astype('bool')
        Seg[:,:,-1][~BG] = 255
        Seg[:,:,1][Seg[:,:,2] == 255] = 255

        IName = str(ResultsDir / str('Seg_' + '%02d' % (i) + '_RFm.png'))
        Figure, Axis = plt.subplots(1,1)
        Axis.imshow(ROIs[i])
        Axis.imshow(Seg.astype('uint8'), alpha=0.5)
        Axis.axis('off')
        plt.subplots_adjust(0,0,1,1)
        plt.savefig(IName)
        plt.close(Figure)

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
