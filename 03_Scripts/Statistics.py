#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script perform statistical analysis of
    automatic segmentation results

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: September 2023
    """

#%% Imports
# Modules import

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from Utils import Time
from pathlib import Path
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap
from scipy.stats.distributions import norm, t
from scipy.stats import pearsonr, shapiro, kstest
from pylatex import Document, Section, NoEscape, Command, Package


#%% Functions
# Define functions

def Histogram(Array:np.array, FigName:str, Labels=[], Bins=20) -> None:

    """
    Plot data histogram along with kernel density and
    corresponding normal distribution to assess data
    normality
    """

    # Compute data values
    X = pd.DataFrame(Array, dtype='float')
    SortedValues = np.sort(X.T.values)[0]
    N = len(X)
    X_Bar = X.mean()
    S_X = np.std(X, ddof=1)

    # Figure plotting
    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)

    # Histogram
    Histogram, Edges = np.histogram(X, bins=Bins, density=True)
    Width = 0.9 * (Edges[1] - Edges[0])
    Center = (Edges[:-1] + Edges[1:]) / 2
    Axes.bar(Center, Histogram, align='center', width=Width,
                edgecolor=(0,0,0), color=(1, 1, 1, 0), label='Histogram')

    # Density distribution
    KernelEstimator = np.zeros(N)
    NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
    DataIQR = np.abs(X.quantile(0.75)) - np.abs(X.quantile(0.25))
    KernelHalfWidth = 0.9 * N ** (-1 / 5) * min(np.abs([S_X, DataIQR / NormalIQR]))
    for Value in SortedValues:
        KernelEstimator += norm.pdf(SortedValues - Value, 0, KernelHalfWidth * 2)
    KernelEstimator = KernelEstimator / N

    Axes.plot(SortedValues, KernelEstimator, color=(1,0,0), label='Kernel density')

    # Corresponding normal distribution
    TheoreticalDistribution = norm.pdf(SortedValues, X_Bar, S_X)
    Axes.plot(SortedValues, TheoreticalDistribution, linestyle='--',
                color=(0,0,1), label='Normal distribution')
    
    if len(Labels) > 0:
        plt.xlabel(Labels[0])
        plt.ylabel(Labels[1])

    plt.legend(loc='best')
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=100)
    plt.close(Figure)

    return

def QQPlot(Array:np.array, FigName:str, Alpha_CI=0.95) -> float:

    """
    Show quantile-quantile plot
    Add Shapiro-wilk test p-value to estimate
    data normality distribution assumption
    Based on: https://www.tjmahr.com/quantile-quantile-plots-from-scratch/
    Itself based on Fox book: Fox, J. (2015)
    Applied Regression Analysis and Generalized Linear Models.
    Sage Publications, Thousand Oaks, California.
    """

    # Shapiro-Wilk test for normality
    W, p = shapiro(Array)

    # Data analysis
    DataValues = pd.DataFrame(Array, dtype='float')
    N = len(DataValues)
    X_Bar = np.mean(DataValues, axis=0)
    S_X = np.std(DataValues,ddof=1)

    # Sort data to get the rank
    Data_Sorted = DataValues.sort_values(0)
    Data_Sorted = np.array(Data_Sorted).ravel()

    # Compute quantiles
    EmpiricalQuantiles = np.arange(0.5, N + 0.5) / N
    TheoreticalQuantiles = norm.ppf(EmpiricalQuantiles, X_Bar, S_X)
    ZQuantiles = norm.ppf(EmpiricalQuantiles,0,1)

    # Compute data variance
    DataIQR = np.quantile(DataValues, 0.75) - np.quantile(DataValues, 0.25)
    NormalIQR = np.sum(np.abs(norm.ppf(np.array([0.25, 0.75]), 0, 1)))
    Variance = DataIQR / NormalIQR
    Z_Space = np.linspace(min(ZQuantiles), max(ZQuantiles), 100)
    Variance_Line = Z_Space * Variance + np.median(DataValues)

    # Compute alpha confidence interval (CI)
    Z_SE = np.sqrt(norm.cdf(Z_Space) * (1 - norm.cdf(Z_Space)) / N) / norm.pdf(Z_Space)
    Data_SE = Z_SE * Variance
    Z_CI_Quantile = norm.ppf(np.array([(1 - Alpha_CI) / 2]), 0, 1)

    # Create point in the data space
    Data_Space = np.linspace(min(TheoreticalQuantiles), max(TheoreticalQuantiles), 100)

    # QQPlot
    BorderSpace = max(0.05*abs(Data_Sorted.min()), 0.05*abs(Data_Sorted.max()))
    Y_Min = Data_Sorted.min() - BorderSpace
    Y_Max = Data_Sorted.max() + BorderSpace

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.plot(Data_Space, Variance_Line, linestyle='--', color=(1, 0, 0), label='Variance :' + str(format(np.round(Variance, 2),'.2f')))
    Axes.plot(Data_Space, Variance_Line + Z_CI_Quantile * Data_SE, linestyle='--', color=(0, 0, 1), label=str(int(100*Alpha_CI)) + '% CI')
    Axes.plot(Data_Space, Variance_Line - Z_CI_Quantile * Data_SE, linestyle='--', color=(0, 0, 1))
    Axes.plot(TheoreticalQuantiles, Data_Sorted, linestyle='none', marker='o', mew=0.5, fillstyle='none', color=(0, 0, 0))
    Axes.text(0.05,0.9,'Shapiro-Wilk p-value: ' + str(round(p,3)),transform=Axes.transAxes)
    plt.xlabel('Theoretical quantiles (-)')
    plt.ylabel('Empirical quantiles (-)')
    plt.ylim([Y_Min, Y_Max])
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), prop={'size':10})
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=100)
    plt.close(Figure)

    return p

def CDFPlot(Array:np.array, FigName:str, Xlabel:str) -> float:

    """
    Plot Empirical cumulative distribution function of
    given array and theorical cumulative distribution
    function of normal distribution. Adds Kolmogorov-Smirnoff
    test p-value to assess data normality ditribution assumption
    """

    # Kolmogorov-Smirnoff test for normality
    KS, p = kstest(Array)

    # Data analysis
    DataValues = pd.DataFrame(Array, dtype='float')
    N = len(DataValues)
    X_Bar = np.mean(DataValues, axis=0)
    S_X = np.std(DataValues,ddof=1)

    # Sort data to get the rank
    Data_Sorted = DataValues.sort_values(0)
    Data_Sorted = np.array(Data_Sorted).ravel()

    # Compute quantiles
    EmpiricalQuantiles = np.arange(0.5, N + 0.5) / N
    Z = (Data_Sorted - X_Bar) / S_X
    TheoreticalQuantiles = norm.cdf(Z)

    Figure, Axes = plt.subplots(1, 1, figsize=(5.5, 4.5), dpi=100)
    Axes.plot(Data_Sorted,EmpiricalQuantiles, linestyle='none', marker='o', mew=0.5, fillstyle='none', color=(0, 0, 1), label='Data Distribution')
    Axes.plot(Data_Sorted,TheoreticalQuantiles, linestyle='--', color=(1, 0, 0), label='Normal Distribution')
    Axes.text(0.05,0.9,'Kolmogorov-Smirnoff p-value: ' + str(round(p,3)),transform=Axes.transAxes)
    plt.xlabel(Xlabel)
    plt.ylabel('Quantile (-)')
    plt.legend(loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.15), prop={'size':10})
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=100)
    plt.close(Figure)

    return p

def PlotOLS(Data:pd.DataFrame, LM:smf.ols, FigName:str,
            Xlabel='X', Ylabel='Y', Alpha_CI=0.95) -> None:
    
    """
    Function used to plot ordinary least square results
    Compute CI bands based on FOX 2017

    Only implemented for 2 levels LME with nested random intercepts
    """

    # Create X values for confidence interval lines
    Min = Data['X'].min()
    Max = Data['X'].max()
    Range = np.linspace(Min, Max, len(Data))

    # Get corresponding fitted values and CI interval
    Y_Fit = LM.params[0] + Range * LM.params[1]
    Alpha = t.interval(Alpha_CI, len(Data) - len(LM.params) - 1)

    # Residual sum of squares
    RSS = np.sum(LM.resid ** 2)

    # Standard error of the estimate
    SE = np.sqrt(RSS / LM.df_resid)

    # Compute corresponding CI lines
    C = np.matrix(LM.cov_params())
    X = np.matrix([np.ones(len(Data)), np.linspace(Min, Max, len(Data))]).T
    
    if C.shape[0] > len(LM.params):
        C = C[:len(LM.params),:len(LM.params)]

    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))

    CI_Line_u = Y_Fit + Alpha[0] * SE * B_0
    CI_Line_o = Y_Fit + Alpha[1] * SE * B_0

    # Plot and save results
    Figure, Axis = plt.subplots(1,1, figsize=(5,5))
    Axis.scatter(Data['X'], Data['Y'], c=Data['Donor'], marker='o', cmap='winter')
    Axis.fill_between(np.linspace(Min, Max, len(Data)), CI_Line_o, CI_Line_u, color=(0,0,0,0.25), edgecolor='none')
    Axis.plot(Range, Y_Fit, color=(1,0,0))
    Axis.set_ylabel(Ylabel)
    Axis.set_xlabel(Xlabel)
    if 'Sex' in Ylabel:
        Axis.set_yticks([0, 1],['M', 'F'])
    if 'Sex' in Xlabel:
        Axis.set_xticks([0, 1],['M', 'F'])
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02)
    plt.close()

    return

def PlotMixedLM(Data:pd.DataFrame, LME:smf.mixedlm, FigName:str,
                Xlabel='X', Ylabel='Y', Alpha_CI=0.95) -> None:
    
    """
    Function used to plot mixed linear model results
    Plotting based on: https://www.azandisresearch.com/2022/12/31/visualize-mixed-effect-regressions-in-r-with-ggplot2/
    As bootstrap is expensive for CI band computation, compute
    CI bands based on FOX 2017

    Only implemented for 2 levels LME with nested random intercepts
    """

    # Compute conditional residuals
    Data['CR'] = LME.params[0] + Data['X']*LME.params[1] + LME.resid

    # Create X values for confidence interval lines
    Min = Data['X'].min()
    Max = Data['X'].max()
    Range = np.linspace(Min, Max, len(Data))

    # Get corresponding fitted values and CI interval
    Y_Fit = LME.params[0] + Range * LME.params[1]
    Alpha = t.interval(Alpha_CI, len(Data) - len(LME.fe_params) - 1)

    # Residual sum of squares
    RSS = np.sum(LME.resid ** 2)

    # Standard error of the estimate
    SE = np.sqrt(RSS / LME.df_resid)

    # Compute corresponding CI lines
    C = np.matrix(LME.cov_params())
    X = np.matrix([np.ones(len(Data)),np.linspace(Min, Max, len(Data))]).T
    
    if C.shape[0] > len(LME.fe_params):
        C = C[:len(LME.fe_params),:len(LME.fe_params)]

    B_0 = np.sqrt(np.diag(np.abs(X * C * X.T)))

    CI_Line_u = Y_Fit + Alpha[0] * SE * B_0
    CI_Line_o = Y_Fit + Alpha[1] * SE * B_0

    # # Plot and save results
    # Figure, Axis = plt.subplots(1,2, sharex=True, sharey=True, figsize=(10,5))
    # Axis[0].scatter(Data['X'], Data['Y'], c=Data['Donor'],
    #                 marker='o', cmap='winter')
    # Axis[0].fill_between(np.linspace(Min, Max, len(Data)), CI_Line_o, CI_Line_u, color=(0,0,0,0.25), edgecolor='none')
    # Axis[0].plot(Range, Y_Fit, color=(1,0,0))
    # Axis[1].scatter(Data['X'], Data['CR'], c=Data['Donor'],
    #                 marker='o', cmap='winter')
    # Axis[1].fill_between(np.linspace(Min, Max, len(Data)), CI_Line_o, CI_Line_u, color=(0,0,0,0.25), edgecolor='none')
    # Axis[1].plot(Range, Y_Fit, color=(1,0,0))
    # Axis[0].set_ylabel(Ylabel)
    # Axis[0].set_ylim([Data['Y'].min() - 0.05*abs(Data['Y'].min()),
    #                   Data['Y'].max() + 0.05*abs(Data['Y'].max())])
    # for i in range(2):
    #     Axis[i].set_xlabel(Xlabel)
    # Axis[0].set_title('Raw Data')
    # Axis[1].set_title('Conditional Residuals')
    # plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02)
    # plt.close()

    # Plot and save results
    Figure, Axis = plt.subplots(1,1, figsize=(6,5))
    Axis.scatter(Data['X'], Data['Y'], c=Data['Donor'],
                    marker='o', cmap='winter')
    Axis.fill_between(np.linspace(Min, Max, len(Data)), CI_Line_o, CI_Line_u, color=(0,0,0,0.25), edgecolor='none')
    Axis.plot(Range, Y_Fit, color=(1,0,0))
    Axis.set_ylabel(Ylabel)
    Axis.set_ylim([Data['Y'].min() - 0.05*abs(Data['Y'].min()),
                      Data['Y'].max() + 0.05*abs(Data['Y'].max())])
    Axis.set_xlabel(Xlabel)
    # Axis.set_title('Raw Data')
    # plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02)
    plt.show()

    return

def BoxPlot(ArraysList:list, FigName:str, Labels=['', 'Y'],
            SetsLabels=None, Vertical=True) -> None:
    
    """
    Save boxplot of a list of arrays
    Used for assessment on random effects and residuals
    """

    Width = 2.5 + len(ArraysList)
    Figure, Axis = plt.subplots(1,1, dpi=100, figsize=(Width,4.5))

    for i, Array in enumerate(ArraysList):
        RandPos = np.random.normal(i,0.02,len(Array))

        Axis.boxplot(Array, vert=Vertical, widths=0.35,
                    showmeans=True,meanline=True,
                    showfliers=False, positions=[i],
                    capprops=dict(color=(0,0,0)),
                    boxprops=dict(color=(0,0,0)),
                    whiskerprops=dict(color=(0,0,0),linestyle='--'),
                    medianprops=dict(color=(0,1,0)),
                    meanprops=dict(color=(0,0,1)))
        Axis.plot(RandPos - RandPos.mean() + i, Array, linestyle='none',
                    marker='o',fillstyle='none', color=(1,0,0), ms=5)
    
    Axis.plot([],linestyle='none',marker='o',fillstyle='none', color=(1,0,0), label='Data')
    Axis.plot([],color=(0,0,1), label='Mean', linestyle='--')
    Axis.plot([],color=(0,1,0), label='Median')
    Axis.set_xlabel(Labels[0])
    Axis.set_ylabel(Labels[1])

    if SetsLabels:
        Axis.set_xticks(np.arange(len(SetsLabels)))
        Axis.set_xticklabels(SetsLabels, rotation=0)
    else:
        Axis.set_xticks([])
    
    plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.125))
    plt.subplots_adjust(left=0.25, right=0.75)
    plt.savefig(FigName, bbox_inches='tight', pad_inches=0.02, dpi=100)
    plt.close()

    return

def Table(Results:dict):

    # Define text parts
    Start = r'''
\begin{center}
\noindent\begin{tabularx}{\linewidth}{|cc|c|cc|cc|}
\toprule
    X & Y & Intercept & Coefficient & P$> |$z$|$ & Donor $\sigma^2$ & Sample $\sigma^2$ \\
\midrule
'''

    Core = '''
{X} & {Y} & {I} [{Ib}, {Iu}] & {C} [{Cb}, {Cu}] & {p} & {D} & {S}  '''

    End = r'''
\bottomrule
\end{tabularx}
\end{center}
'''
        

    # Get dictionnary indices
    XC, YC = [], []
    for k in Results.keys():
        if k[0] not in XC:
            XC.append(k[0])
        if k[1] not in YC:
            YC.append(k[1])

    # Build text
    Text = Start
    for iX, Xc in enumerate(XC):
        for iY, Yc in enumerate(YC):
            if iX > iY:

                # Get coefficients
                I, C, D, S = Results[(Xc, Yc)].params
                p = Results[(Xc, Yc)].pvalues[1]

                # get confidence intervals
                CI = Results[(Xc, Yc)].conf_int()
                Ib = CI.loc['Intercept',0]
                Iu = CI.loc['Intercept',1]
                Cb = CI.loc['X',0]
                Cu = CI.loc['X',1]

                # Create label
                XLabel = (Xc[0][0] + ' - ' + Xc[1]).replace('%','\%')
                YLabel = (Yc[0][0] + ' - ' + Yc[1]).replace('%','\%')

                # Round results according to their values
                Variables = [I, C, D, S, Ib, Iu, Cb, Cu]
                for i, V in enumerate(Variables[:4]):

                    Log10 = np.log10(abs(V))
                    Sign = Log10 / abs(Log10)
                    if Sign < 0:
                        R = int(np.ceil(abs(Log10)) + 2)
                        V = round(V,R)
                    else:
                        R = -int(np.floor(abs(Log10)) - 2)
                        V = int(round(V,R))

                    # If variable is too long, change to scientific notation
                    L = len(f'{V}')
                    if L > 6:
                        Variables[i] = f'{V:.2e}'
                        if i in [0, 1]:
                            V = round(Variables[2*i+4], R)
                            Variables[2*i+4] = f'{V:.2e}'
                            V = round(Variables[2*i+5], R)
                            Variables[2*i+5] = f'{V:.2e}'

                    else:
                        Variables[i] = f'{V}'
                        if i in [0, 1]:
                            V = round(Variables[2*i+4], R)
                            Variables[2*i+4] = f'{V}'
                            V = round(Variables[2*i+5], R)
                            Variables[2*i+5] = f'{V}'

                I, C, D, S, Ib, Iu, Cb, Cu = Variables

                Context = {'X': XLabel,
                           'Y': YLabel,
                           'I':I,
                           'Ib':Ib,
                           'Iu':Iu,
                           'C':C,
                           'Cb':Cb,
                           'Cu':Cu,
                           'p': round(p,3),
                           'D': D,
                           'S': S}
                
                Text += Core.format(**Context) + r'\\'
    Text += End

    return Text

#%% Main
# Main part

def Main(Alpha=0.05):

    # Record time elapsed
    Time.Process(1, 'Segmentation stats')

    # Set paths
    MainDir = Path(__file__).parent / '..'
    DataDir = MainDir / '01_Data'
    ResDir = MainDir / '03_Results'
    StatDir = ResDir / 'Statistics' 
    Path.mkdir(StatDir, exist_ok=True)

    # Load data
    SampleData = pd.read_excel(DataDir / 'Data.xlsx', index_col=[0,1])
    Data = pd.read_csv(ResDir / 'SegmentationData.csv', header=[0,1])
    
    # Set first 3 columns as multiindex
    Indices = (R for R in Data[Data.columns[:3]].values)
    Data.index = pd.MultiIndex.from_tuples(Indices)
    Data.index.names = ('Donor', 'Side', 'ROI')

    # Drop columns used and columns and nan
    Data = Data.drop([(C) for C in Data.columns[:3]], axis=1)
    Data = Data.dropna(axis=0, how='all')
    Data = Data.dropna(axis=1, how='all')

    # Replace sex letter by numeric dummy variables
    Dict = {'M':0,'F':1}
    SampleData['Sex (-)'] = SampleData['Sex (-)'].replace(Dict)

    # Add sex and age data
    Data[('Donor','Sex (-)')] = np.nan
    Data[('Donor','Age (year)')] = np.nan
    for Idx in Data.index:
        Donor = int(Idx[0][-2:])
        Side = Idx[1][0]
        Index = (Donor,Side)
        if Index in SampleData.index:
            Data.loc[Idx,('Donor','Sex (-)')] = SampleData.loc[Index,'Sex (-)']
            Data.loc[Idx,('Donor','Age (year)')] = SampleData.loc[Index,'Age (year)']
    Data = Data.dropna(axis=0, how='any')

    # Perform LME for each variables
    Results = {}
    for Xc in Data.columns:
        for Yc in Data.columns:

            # Set axis labels and figure name
            FName = Xc[0] + Xc[1].split()[0] + '_'
            FName += Yc[0] + Yc[1].split()[0]
            FName = StatDir / (FName + '.png')
            
            Xlabel = Xc[0] + ' - ' + Xc[1]
            Ylabel = Yc[0] + ' - ' + Yc[1]

            # Build data frame for linear regression
            X = Data[Xc].values.astype('float')
            Y = Data[Yc].values.astype('float')
            DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])
            
            # Add donor group and sample groups
            DataFrame['Donor'] = [int(D[-2:]) for D in Data.index.get_level_values('Donor')]
            DataFrame['Sample'] = Data.index.get_level_values('Side')

            if Xc == Yc:
                LM = smf.ols('Y ~ X', data=DataFrame).fit()
            else:
                # 2 levels LME with random intercepts
                LM = smf.mixedlm('Y ~ X',
                                data=DataFrame, groups=DataFrame['Donor'],
                                re_formula='1', vc_formula={'Sample': '0 + Sample'}
                                ).fit(reml=False, method=['lbfgs'])
            
            # Plot resuts
            if 'Sex (-)' == Xc[1]:
                BoxPlot([Y[X==0], Y[X==1]], FName, [Xlabel, Ylabel], ['M','F'])
            elif Xc == Yc:
                PlotOLS(DataFrame,LM, FName, Xlabel, Ylabel)            
            else:
                PlotMixedLM(DataFrame,LM, FName, Xlabel, Ylabel)
            
            # Store results in dictionary
            Results[(Xc,Yc)] = LM

    # Save results
    with open(StatDir / 'Results.pkl', 'wb') as F:
        pickle.dump(Results, F)

    # Load results
    with open(StatDir / 'Results.pkl', 'rb') as F:
        Results = pickle.load(F)

    # Build p-values matrix
    Corr = np.zeros((len(Data.columns), len(Data.columns)))
    for iX, Xc in enumerate(Data.columns):
        for iY, Yc in enumerate(Data.columns):
            if iX <= iY:
                Corr[iX,iY] = np.nan
            else:
                LME = Results[(Xc, Yc)]
                Corr[iX,iY] = round(LME.pvalues[1],3)

    # Categorise p-values
    Cat = Corr.copy()
    Cat[Cat >= Alpha] = 4
    Cat[Cat < 0.001] = 1
    Cat[Cat < 0.01] = 2
    Cat[Cat < 0.05] = 3

    # Plot p-values categories
    Labels = [C[0] + '\n' + C[1] for C in Data.columns]
    Figure, Axis = plt.subplots(1,1, figsize=(9,12))
    Im = Axis.matshow(Cat, vmin=0, vmax=4.5, cmap='binary')
    Axis.xaxis.set_ticks_position('bottom')
    Axis.set_xticks(np.arange(len(Corr))-0.5, minor=True)
    Axis.set_xticks(np.arange(len(Corr)))
    Axis.set_xticklabels(Labels, ha='center', rotation=90)
    Axis.set_yticks(np.arange(len(Corr))-0.5, minor=True)
    Axis.set_yticks(np.arange(len(Corr)))
    Axis.set_yticklabels(Labels)
    Axis.grid(which='minor', color=(1,1,1), linestyle='-', linewidth=2)
    Cb = plt.colorbar(Im, ticks=[1, 2, 3, 4], fraction=0.046, pad=0.04, values=np.linspace(1,4,100))
    Cb.ax.set_yticklabels(['<0.001', '<0.01', '<0.05','$\geq$0.05'])
    # Figure.savefig(StatDir / 'Pvalues.png', dpi=Figure.dpi, bbox_inches='tight')
    plt.show(Figure)

    # Save results in tex file
    GeOptions = {'paperheight':'270mm',
                 'paperwidth':'250mm',
                 'left':'5mm',
                 'top':'5mm',
                 'right':'5mm',
                 'bottom':'5mm'}
    Doc = Document(default_filepath=str(StatDir / 'Results'),
                   documentclass='article',
                   geometry_options=GeOptions)
    Text = Table(Results)
    Doc.append(NoEscape(Text))
    Doc.packages.append(Package('tabularx'))
    Doc.packages.append(Package('geometry',['landscape']))
    Doc.packages.append(Package('booktabs'))
    Doc.generate_pdf(clean_tex=False)


    # # Build data frame to collect stats
    # StatCols = ['Intercept', 'X', 'p', 'Donor', 'Sample']
    # Stats = pd.DataFrame(index=Data.columns, columns=StatCols)

    # # Update time spend
    # Time.Update(1 / 8)
    # Nc = len(Data.columns)
    # Nt = len(SegData.columns)

    # # Analyse segment selative densities
    # for i, Tissue in enumerate(SegData.columns):

    #     # Log transformation of data to meet normality assumption
    #     # Data[Tissue] = np.log(Data[Tissue].values.astype(float))

    #     # Create folder to store tissue results
    #     Folder = StatDir / Tissue.split(' (')[0]
    #     os.makedirs(Folder, exist_ok=True)

    #     # Independent variable
    #     Y = Data['Final',Tissue].values.astype('float')

    #     # Normality of independant variable
    #     Histogram(Y, str(Folder / 'Density_Hist.png'),
    #               Labels=['Density (%)','Relative count (-)'])
    #     Shapiro = QQPlot(Y, str(Folder / 'Density_QQ.png'))

    #     # If p < 0.05, null hypothesis of normal distribution
    #     # is rejected -> Normal assumtpion of LME not satisfied
    #     if Shapiro < Alpha:
    #         print('\n' + Tissue.split(' (')[0] + ' not normally distributed')
        
    #     for j, (M,T) in enumerate(Data.columns):

    #         # Avoid auto correlation
    #         if M == Methods[-1] and T == Tissue:
    #             pass

    #         # If methods is different, do not analyse correlation
    #         # with different tissue
    #         elif M in Methods[:-1] and T != Tissue:
    #             pass

    #         else:
    #             # Split variable name to build plot x label
    #             Split = T.split(' / ')
    #             Label = Split[0] + ' ('
    #             for S in Split[1:]:
    #                 Label += S.split()[0] + '/'
    #             Label = Label[:-1] + ')'

    #             # Build data frame with dependent and independent variable
    #             X = Data[M,T].values.astype('float')
    #             DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])

    #             # Add donor group and sample groups
    #             DataFrame['Donor'] = [int(D[-2:]) for D in Data.index.get_level_values('Donor')]
    #             DataFrame['Sample'] = Data.index.get_level_values('Side')

    #             # If same tissue, OLS regression
    #             if T == Tissue:
    #                 LM = smf.ols('Y ~ X', data=DataFrame).fit()

    #             else:
    #                 # Fit 2 Levels LME
    #                 LM = smf.mixedlm('Y ~ X',
    #                                 data=DataFrame, groups=DataFrame['Donor'],
    #                                 re_formula='1', vc_formula={'Sample': '0 + Sample'}
    #                                 ).fit(reml=False, method=['lbfgs'])

    #             # Set file name
    #             if M:
    #                 FName = M
    #             else:
    #                 FName = T.split(' (')[0]

    #             # Check data normality assumptions
    #             Histogram(X, str(Folder / (FName + '_Hist.png')),
    #                         Labels=[Label,'Relative count (-)'])
    #             Shapiro = QQPlot(X, str(Folder / (FName + '_QQ.png')))

    #             if T == Tissue:
    #                 PlotOLS(DataFrame, LM, str(Folder / (FName + '_OLS.png')),
    #                         Xlabel=Label, Ylabel='Density (%)')
    #             else:
    #                 # Plot 2 levels LME
    #                 PlotMixedLM(DataFrame, LM, str(Folder / (FName + '_LME.png')),
    #                             Xlabel=Label, Ylabel='Density (%)')

    #                 # Check random effects assumptions
    #                 RE = pd.DataFrame(LM.random_effects).T
    #                 RE.columns = ['Group','Left','Right']
    #                 BoxPlot([RE['Group'], RE['Left'].dropna(), RE['Right'].dropna()],
    #                         str(Folder / (FName + '_RE.png')),
    #                         SetsLabels=['Donor','Left','Right'],
    #                         Labels=['','Random Effects'])

    #                 Shapiro = QQPlot(RE['Group'].values, str(Folder / (FName + '_Group_QQ.png')))
    #                 Shapiro = QQPlot(RE['Left'].dropna().values, str(Folder / (FName + '_Left_QQ.png')))
    #                 Shapiro = QQPlot(RE['Right'].dropna().values, str(Folder / (FName + '_Right_QQ.png')))

    #             # Check residuals
    #             BoxPlot([LM.resid],
    #                     str(Folder / (FName + '_Residuals.png')),
    #                     Labels=['','Residuals'])
    #             Shapiro = QQPlot(LM.resid.values, str(Folder / (FName + '_Residuals_QQ.png')))

    #             # Store values
    #             Stats.loc[(M, T), 'X'] = LM.params[1]
    #             Stats.loc[(M, T), '0.025'] = LM.conf_int().loc['X',0]
    #             Stats.loc[(M, T), '0.975'] = LM.conf_int().loc['X',1]
    #             Stats.loc[(M, T), 'p'] = LM.pvalues[1]

    #         # Print elapsed time
    #         Time.Update(((i+1) * Nc / Nt + (j+1) / Nc) / (Nc+1) * 7/8 + 1/8)

    #     # Save data and generate a new one
    #     Stats = Stats.dropna()
    #     Stats.to_csv(ResDir / (Tissue.replace(' ','_')[:-4] + '_Statistics.csv'))
    #     Stats = pd.DataFrame(index=Data.columns, columns=StatCols)

    # # Analyse morphometry correlations of Haversian canals and osteocytes
    # Labels = ['Haversian canals', 'Osteocytes', 'CementLines']
    # Data = pd.DataFrame(columns=['Parameter','Y','X','Coef','0.025','0.075','p'],
    #                     index=np.arange(1,9))
    # i = 0
    # for Stats, Label in zip([HcStats, OsStats, ClStats], Labels):
    #     if Label in Labels[:-1]:
    #         Parameters = ['Area (-)', 'Number (-)']
    #     else:
    #         Parameters = ['Thickness (px)']
    #     for Par in Parameters:
    #         for Var in ['Sex (-)', 'Age (year)']:
    #             Y = Stats[Par].values
    #             X = Stats[Var].values
    #             DataFrame = pd.DataFrame(np.vstack([X,Y]).T,columns=['X','Y'])

    #             # Add donor group and sample groups
    #             DataFrame['Donor'] = [int(D[-2:]) for D in Stats.index.get_level_values(0)]
    #             DataFrame['Sample'] = Stats.index.get_level_values(1)

    #             LME = smf.mixedlm('Y ~ X',
    #                             data=DataFrame, groups=DataFrame['Donor'],
    #                             re_formula='1', vc_formula={'Sample': '0 + Sample'}
    #                             ).fit(reml=False, method=['lbfgs'])

    #             # Store in data frame
    #             i += 1
    #             Data.loc[i] = [Par,
    #                            Label, Var, LME.params[1],
    #                            LME.conf_int().loc['X',0],
    #                            LME.conf_int().loc['X',1],
    #                            LME.pvalues[1]]
    # Data.to_csv(ResDir / 'MorphoCorr.csv')


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
