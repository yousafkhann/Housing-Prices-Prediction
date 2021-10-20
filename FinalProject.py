from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
import pandas as pd
import scipy.spatial as sci
import timeit
import math
import numpy as np
from sklearn import model_selection
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")

    #demonstrateHelpers(trainDF)
    
    #print(corrTest(trainDF))

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    
    doExperiment(trainInput, trainOutput, predictors)
    
    #hypParamTest(trainInput, trainOutput, predictors)
    
    #doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    

    
    
    

    
# ===============================================================================
def readData(numRows = None):
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    
    outputCol = ['SalePrice']
    
    return trainDF, testDF, outputCol
    
def corrTest(df):
    
    colNames = ['MSSubClass', 'LotArea', 'LotFrontage', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', '2ndFlrSF', '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'YrSold']
    salePrice = df.iloc[:,-1]
    #data, test, output = readData()
    
    corrDF = df.loc[:, colNames]
    
    corrFigures = corrDF.apply(lambda col: col.corr(salePrice) ,axis =0)
    
    return corrFigures

 
def hypParamTest(trainInput, trainOutput, predictors):
    '''
    depthList = pd.Series([2,2.5,3,3.5,4,4.5,5])
    
    accuracies = depthList.map(lambda x: (model_selection.cross_val_score(GradientBoostingRegressor(max_depth = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    
    plt.plot(depthList, accuracies)
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    
    
    print("Most Efficient: ", depthList.loc[accuracies.idxmax()])
    '''
    
    '''
    alphaList = pd.Series([50,100,150,200,250,300])
    
    accuracies = alphaList.map(lambda x: (model_selection.cross_val_score(Ridge(alpha = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    
    plt.plot(alphaList, accuracies)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    
    
    print("Most Efficient: ", alphaList.loc[accuracies.idxmax()])
    '''
    alphaList = pd.Series([50,100,200,300,400,500])
    
    accuracies = alphaList.map(lambda x: (model_selection.cross_val_score(Lasso(alpha = x), trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')).mean())
    print(accuracies)
    
    
    plt.plot(alphaList, accuracies)
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    
    
    print("Most Efficient: ", alphaList.loc[accuracies.idxmax()])
    

'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)
    
    '''
    We added GradientBoostingRegressor algorithm within the doExperiment Function
    '''
    
    alg1 = GradientBoostingRegressor()
    alg1.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg1, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    
    '''
    We added Ridge Regression
    '''
    alg2 = Ridge(alpha=250)
    alg2.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg2, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())

    '''
    We added Lasso Regression
    '''
    alg3 = Lasso(alpha=800)
    alg3.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg3, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    
    '''
    We added Bagging meta-estimator
    '''
    alg4 = BaggingRegressor(KNeighborsClassifier(), max_samples = 0.85, max_features=0.85)
    alg4.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg4, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2')
    print("Accuracy: ", cvScores.mean())
    
    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    '''
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle
    '''
    
    '''
    WE ARE USING GRADIENTBOOSTINGREGRESSOR AS IT GAVE THE BEST PREDICTIONS FOR OUR ANALYSIS
    '''
    
    alg = GradientBoostingRegressor()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle
# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    fullDF = trainDF
    trainInput = trainDF.iloc[:, :80]
    testInput = testDF.iloc[:, :]
    
   
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    
    
    
    'start preprocessing'
    
    '''
    We ran correlation test with all existing numerical columns and dropped all columns that had correlations between - 0.2
    and +0.3. This is because, we believe that such small correlation is not a good predictor for sales price.
    
    '''
    trainInput = trainInput.drop(['LotArea','OverallCond','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','YrSold'], axis=1)
    testInput = testInput.drop(['LotArea','OverallCond','BsmtUnfSF','BsmtFullBath','BsmtHalfBath','YrSold'], axis=1)
    
    '''
    After inspecting the dataset we noticed that PoolQC, MiscVal attributes were almost entirely 'NA'. Hence, it does not
    provide anything valuable for our analysis. Therefore, we dropped it.
    We also realized that RoofMatl attribute had mostly 'CompShg' values. Following the same logic, it is not useful
    for our comparison. We dropped it.
    '''
    trainInput = trainInput.drop(['PoolQC','MiscVal','RoofMatl'], axis =1)
    testInput = testInput.drop(['PoolQC','MiscVal','RoofMatl'], axis =1)   
    
     #print(indCorr(fullDF, 'PoolQC'))
    
    #print(indCorr(fullDF, 'MiscVal')) faarik
    #print(indCorr(fullDF, 'RoofMatl'))
    #print(indCorr(fullDF, 'Alley'))

    '''
    We classified and checked correlation with SalePrice for Fence, it was too low so we dropped it.
    
    fullDF.loc[:,'Fence'].fillna(0, inplace=True)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 3 if(val=='GdPrv') else val)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 3 if(val=='GdWo') else val)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 2 if(val== 'MnPrv') else val)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 2 if(val== 'MnWw') else val)
    fullDF.loc[:,'Fence'] = fullDF.loc[:,'Fence'].map(lambda val: 0 if(val== 'NA') else val)
    print(indCorr(fullDF, 'Fence')) 
    '''
    trainInput.drop('Fence', axis=1)
    testInput.drop('Fence', axis=1)
    
    
    '''
    MasVnrArea has missing values but has high correlation with SalePrice. We are going to fill the missing ones.
    LotFrontage also has missing values but has high correlation with SalePrice. We are going to fill the missing ones.
    '''
    
    trainInput.loc[:, 'MasVnrArea'].fillna(method='bfill', inplace=True)
    testInput.loc[:,'MasVnrArea'].fillna(method='bfill', inplace=True)
    
    trainInput.loc[:, 'LotFrontage'].fillna(method='bfill', inplace=True)
    testInput.loc[:,'LotFrontage'].fillna(method='bfill', inplace=True)
      
    '''
    Garage Cars and BsmtFinSF1 have missing values, we need to fill them before we can use as predictors.
    '''
    trainInput.loc[:, 'GarageCars'].fillna(trainInput.loc[:, 'GarageCars'].mean(), inplace=True)
    testInput.loc[:, 'GarageCars'].fillna(trainInput.loc[:, 'GarageCars'].mean(), inplace=True)
    
    trainInput.loc[:, 'BsmtFinSF1'].fillna(trainInput.loc[:, 'BsmtFinSF1'].mean(), inplace=True)
    testInput.loc[:, 'BsmtFinSF1'].fillna(trainInput.loc[:, 'BsmtFinSF1'].mean(), inplace=True)
    
    
    

    
    '''
    Column PoolArea has values that are predominately equal 
    to zero. This does not provide a good indication to the algorithm in making predictions. 
    
    '''
    trainInput = trainInput.drop('PoolArea', axis =1)
    testInput = testInput.drop('PoolArea', axis =1)    
    
    '''
    Standardized 
    
    '''
    standardizeCols = ['OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','2ndFlrSF','1stFlrSF','GrLivArea','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','LotFrontage']
    standardize(trainInput, standardizeCols )
    standardize(testInput, standardizeCols)
    
    
    predictors = ['OverallQual', 'YearBuilt','YearRemodAdd','2ndFlrSF','1stFlrSF','GrLivArea','Fireplaces','WoodDeckSF','OpenPorchSF','BsmtCond','YearsOld','GrgYrsOld','BsmtFinType1','OpenPorchSF','HouseStyle', 'ExterQual', 'HeatingQC','BsmtCond', 'YearsOld', 'GrgYrsOld','BsmtFinType1','OpenPorchSF','BsmtFinSF1','MasVnrArea','YearRemodAdd','GarageCars','HouseStyle', 'ExterQual', 'HeatingQC']
    
    
    
    '''
    Day 2
    '''
    '''
    MSZONING
    '''
   
   
    '''
    values = fullDF.loc[:,"MSZoning"].value_counts().values
    mylabels = fullDF.loc[:,"MSZoning"].unique()
    mycolors = ['lightblue', 'lightsteelblue','silver','red','gold']
    myexplode = (0.1,0,0,0,0)
    plt.pie(values, labels = mylabels, autopct='%1.1f%%', startangle = 15, shadow = True, colors = mycolors, explode = myexplode)
    plt.title('MSZoning')
    plt.axis('equal')
    plt.show()
    '''
    
    '''
    After creating a pie chart of MSZONING values, we realized that only .8% of the values are other than RL, RM, and C. Therefore, we are classifying by grouping together some 
    of the other values. Next, we will check correlation between these and Sale Price to see the relevance of MSZoning in our analysis.
    '''
    
    
    
    '''
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 0 if v=="RL" else v)
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 1 if v=="RM" else v)
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 2 if v=="C (all)" else v)
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 3 if v=="FV" else v)
    fullDF.loc[:, "MSZoning"] = fullDF.loc[:, "MSZoning"].map(lambda v: 3 if v=="RH" else v)
    
    print(indCorr(fullDF, 'MSZoning'))
    '''
    
    '''
    After classifying MSZoning (As seen above), we got a corr figure of -0.113, which is not significant enough for our analysis. Thus, we are choosing
    to drop it.
    
    
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 0 if v=="RL" else v)
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 1 if v=="RM" else v)
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 2 if v=="C (all)" else v)
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 3 if v=="FV" else v)
    trainInput.loc[:, "MSZoning"] = trainInput.loc[:, "MSZoning"].map(lambda v: 3 if v=="RH" else v)
    '''
    
    trainInput.drop('MSZoning', axis = 1)
    testInput.drop('MSZoning', axis = 1)
    

    '''
    BSMT Qual and BSMT Condition
    '''
    
    '''
    BsmtQual and BsmtCond have the same variable names. Consequently, we will be classifying them to figure out any correlation between the two
    Both have NaN values, thus we will start by filling them so they are identifiable
    
    
    fullDF.loc[:, 'BsmtQual'] = fullDF.loc[:, 'BsmtQual'].fillna('None')
    fullDF.loc[:, 'BsmtCond'] = fullDF.loc[:, 'BsmtCond'].fillna('None')
    
    values = fullDF.loc[:,"BsmtQual"].value_counts().values
    mylabels = fullDF.loc[:,"BsmtQual"].unique()
    mycolors = ['lightblue', 'lightsteelblue','silver','red','gold']
    myexplode = (0.1,0,0,0,0)
    plt.pie(values, labels = mylabels, autopct='%1.1f%%', startangle = 45, shadow = True, colors = mycolors, explode = myexplode)
    plt.title('BsmtQual')
    plt.axis('equal')
    plt.show()
    
    values = fullDF.loc[:,"BsmtCond"].value_counts().values
    mylabels = fullDF.loc[:,"BsmtCond"].unique()
    mycolors = ['lightblue', 'lightsteelblue','silver','red','gold']
    myexplode = (0.1,0,0,0,0)
    plt.pie(values, labels = mylabels, autopct='%1.1f%%', startangle = 45, shadow = True, colors = mycolors, explode = myexplode)
    plt.title('BsmtCond')
    plt.axis('equal')
    plt.show()
    
    
    Though the variables are the same, there is a disparity in frequency as shown by the charts. Next, we want to do a Correlation test to dig deeper
    Additionally, Fa and None both have very small frequncies in both attributes, thus we are combining them.
    
    
    
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 1 if(val=='TA') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 2 if(val=='Gd') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 3 if(val== 'Ex') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 4 if(val== 'Fa') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 4 if(val==  'None') else val)
    fullDF.loc[:,'BsmtQual'] = fullDF.loc[:,'BsmtQual'].map(lambda val: 5 if(val== 'Po') else val)
    
    

    
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 1 if(val=='TA') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 2 if(val=='Gd') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 3 if(val== 'Ex') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Fa') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 4 if(val==  'None') else val)
    fullDF.loc[:,'BsmtCond'] = fullDF.loc[:,'BsmtCond'].map(lambda val: 5 if(val== 'Po') else val)
    
    
    print(fullDF.loc[:, 'BsmtCond'].corr(fullDF.loc[:,'BsmtQual']))
    
    '''
    
    '''
    The Correlation Test showed a strong correlation between BsmtCond and BsmtQual, we are choosing to stick with BsmtCond, as it increased the accuracy
    of our prediction. Additionally, we are leaving BsmtCond discretized.
    '''

  
    testInput.loc[:, 'BsmtCond'] = testInput.loc[:, 'BsmtCond'].fillna('None')
    
    
    trainInput.loc[:, 'BsmtCond'] = trainInput.loc[:, 'BsmtCond'].fillna('None')
    
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 1 if(val=='TA') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 2 if(val=='Gd') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 3 if(val== 'Ex') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Fa') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val==  'None') else val)
    testInput.loc[:,'BsmtCond'] = testInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Po') else val)
    
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 1 if(val=='TA') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 2 if(val=='Gd') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 3 if(val== 'Ex') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Fa') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val==  'None') else val)
    trainInput.loc[:,'BsmtCond'] = trainInput.loc[:,'BsmtCond'].map(lambda val: 4 if(val== 'Po') else val)
    
   
    
    trainInput.drop('BsmtQual', axis = 1)
    testInput.drop('BsmtQual', axis = 1)
    
    
    
    '''
    
    BsmtFinType1 and BsmtFinType2
    
    Next, we noticed that BsmtFinType1 and BsmtFinType2 shared the exact same variables as well. 
    We will classify both and check correlation to see if both are relevant to our prediction
    We also noticed that both values have significnat missing values, thus we decided to fill them with the mode
    '''
    '''
    
    fullDF.loc[:,'BsmtFinType1'].fillna(fullDF.loc[:,'BsmtFinType1'].mode())
    fullDF.loc[:,'BsmtFinType2'].fillna(fullDF.loc[:,'BsmtFinType2'].mode())
    
    
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 1 if(val=='GLQ') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 2 if(val=='ALQ') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 3 if(val== 'BLQ') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 4 if(val== 'LwQ') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 5 if(val==  'NA') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 6 if(val== 'Rec') else val)
    fullDF.loc[:,'BsmtFinType1'] = fullDF.loc[:,'BsmtFinType1'].map(lambda val: 7 if(val== 'Unf') else val)
    

    
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 1 if(val=='GLQ') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 2 if(val=='ALQ') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 3 if(val== 'BLQ') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 4 if(val== 'LwQ') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 5 if(val==  'NA') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 6 if(val== 'Rec') else val)
    fullDF.loc[:,'BsmtFinType2'] = fullDF.loc[:,'BsmtFinType2'].map(lambda val: 7 if(val== 'Unf') else val)
    
    print(indCorr(fullDF, 'BsmtFinType1'))
    print(indCorr(fullDF, 'BsmtFinType2'))
    '''
    
    '''
    Doing our correlation test, BsmtFinType1 and BsmtFinType2 yielded a -0.2687 and 0.0415 correlation figure respectively. The second one is lower than
    our established threshold. Therefore, we decided to drop BsmtFinType2.
    '''
    
    testInput.loc[:,'BsmtFinType1'].fillna(method='bfill', inplace=True)
    trainInput.loc[:,'BsmtFinType1'].fillna(method='bfill', inplace=True)
    
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 1 if(val=='GLQ') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 2 if(val=='ALQ') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 3 if(val== 'BLQ') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 4 if(val== 'LwQ') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 5 if(val==  'NA') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 6 if(val== 'Rec') else val)
    testInput.loc[:,'BsmtFinType1'] = testInput.loc[:,'BsmtFinType1'].map(lambda val: 7 if(val== 'Unf') else val)
    
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 1 if(val=='GLQ') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 2 if(val=='ALQ') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 3 if(val== 'BLQ') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 4 if(val== 'LwQ') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 5 if(val==  'NA') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 6 if(val== 'Rec') else val)
    trainInput.loc[:,'BsmtFinType1'] = trainInput.loc[:,'BsmtFinType1'].map(lambda val: 7 if(val== 'Unf') else val)
    
    
    trainInput.drop('BsmtFinType2', axis = 1)
    testInput.drop('BsmtFinType2', axis = 1)
    
    '''
    YearBuilt
    '''
    
    '''
    We created a couple of additional columns to further expand and make linear the YearBuilt data.
    The DecOld variable helped categorize the years better to represent a more meaninful comparison for our algorithm
    This led to an improvement in our prediction
    '''
    

    
    trainInput.loc[:,'YearsOld'] = trainInput.loc[:, 'YearBuilt'].map(lambda x: 2021-x)
    #trainInput.loc[:,'DecOld'] = fullDF.loc[:, 'YearsOld'].map(lambda x: x//5)
    
    trainInput.loc[:,'YearsOld'].fillna(method='bfill', inplace=True)
    
    testInput.loc[:,'YearsOld'] = testInput.loc[:, 'YearBuilt'].map(lambda x: 2021-x)
    #testInput.loc[:,'DecOld'] = fullDF.loc[:, 'YearsOld'].map(lambda x: x//5)
    
    testInput.loc[:,'YearsOld'].fillna(method='bfill', inplace=True)
    
    #print(indCorr(fullDF,'YearsOld'))
    #print(indCorr(fullDF,'DecOld'))
    #print(indCorr(fullDF, 'YearBuilt'))
    
    
    '''
    GarageYrBlt
    '''
    '''
    We did the same analysis with GarageYrBlt
    '''
    
    trainInput.loc[:,'GrgYrsOld'] = trainInput.loc[:, 'GarageYrBlt'].map(lambda x: 2021-x)
    #trainInput.loc[:,'GrgDecOld'] = fullDF.loc[:, 'GarageYrBlt'].map(lambda x: x//5)
    

    trainInput.loc[:,'GrgYrsOld'].fillna(method='bfill', inplace=True)
    
    
    testInput.loc[:,'GrgYrsOld'] = fullDF.loc[:, 'GarageYrBlt'].map(lambda x: 2021-x)
    #testInput.loc[:,'GrgDecOld'] = fullDF.loc[:, 'GarageYrBlt'].map(lambda x: x//5)
    
    
    
    testInput.loc[:,'GrgYrsOld'].fillna(method='bfill', inplace=True)
    
    
    #print(testInput.loc[:,'GrgYrsOld'])
    #print(indCorr(fullDF,'GarageYrBlt'))
    #print(indCorr(fullDF,'GrgDecOld'))
    #print(indCorr(fullDF, 'GrgYrsOld'))
    
    '''
    CentralAir
    '''
    
    '''
    CentralAir attribute only had 2 values -> we want to discretize it and check correlation with SalePrice.
    During the correlation test, the correlation for Central Air was less than +0.3. Thus, it does not meet our requirements and we will drop it.
    '''  
    
    '''
    
    fullDF.loc[:,'CentralAir'] = fullDF.loc[:,'CentralAir'].map(lambda val: 1 if(val=='Y') else val)
    fullDF.loc[:,'CentralAir'] = fullDF.loc[:,'CentralAir'].map(lambda val: 0 if(val=='N') else val)
    
    print(indCorr(fullDF, 'CentralAir'))
    '''
    
    trainInput.drop('CentralAir', axis = 1)
    testInput.drop('CentralAir', axis = 1)
    
    
    '''
    The dataset has 4 porch related attributes: OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch.
    We want to analyze these.
    '''
    '''
    print(indCorr(fullDF, 'OpenPorchSF'))
    print(indCorr(fullDF, 'EnclosedPorch'))
    print(indCorr(fullDF, '3SsnPorch'))
    print(indCorr(fullDF, 'ScreenPorch'))
    '''
    '''
    Next, we decided to add the attributes together, to create a total porch area variable.
    '''
    
    #fullDF.loc[:, 'ttlPorch'] = fullDF.loc[:, 'OpenPorchSF']+fullDF.loc[:, 'EnclosedPorch']+fullDF.loc[:, 'ScreenPorch']
    #print(indCorr(fullDF, 'ttlPorch'))
    
    #testInput.loc[:, 'ttlPorch'] = testInput.loc[:, 'OpenPorchSF']+testInput.loc[:, 'EnclosedPorch']+testInput.loc[:, 'ScreenPorch']
    #trainInput.loc[:, 'ttlPorch'] = trainInput.loc[:, 'OpenPorchSF']+trainInput.loc[:, 'EnclosedPorch']+trainInput.loc[:, 'ScreenPorch']
    
    '''
    The new attribute decreased our accuracy, and had a low correlation score. We will only use OpenPorchSF, as it has a correlation that is greater than
    +0.3
    '''
    trainInput = trainInput.drop(['3SsnPorch','EnclosedPorch','ScreenPorch'], axis =1)
    testInput = testInput.drop(['3SsnPorch','EnclosedPorch','ScreenPorch'], axis =1)    
    
    '''
    Next, we are looking to classify ordinal values in the Condition1 and Condition2 attributes.
    We noticed that some values relate to roads, some to railways, and some to positive features.
    We are going to classify the values accordingly.
    '''
    '''
    
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 0 if(val=='Artery') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 0 if(val=='Feedr') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 1 if(val=='RRNn') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 1 if(val=='RRAn') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 1 if(val=='RRNe') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 1 if(val=='RRAe') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 2 if(val=='Norm') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 3 if(val=='PosN') else val)
    fullDF.loc[:,'Condition1'] = fullDF.loc[:,'Condition1'].map(lambda val: 3 if(val=='PosA') else val)
    
    print(indCorr(fullDF, 'Condition1'))
    
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 0 if(val=='Artery') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 0 if(val=='Feedr') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 1 if(val=='RRNn') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 1 if(val=='RRAn') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 1 if(val=='RRNe') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 1 if(val=='RRAe') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 2 if(val=='Norm') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 3 if(val=='PosN') else val)
    fullDF.loc[:,'Condition2'] = fullDF.loc[:,'Condition2'].map(lambda val: 3 if(val=='PosA') else val)
    
    print(indCorr(fullDF, 'Condition2'))
    
    fullDF.loc[:,'CondSum'] =fullDF.loc[:,'Condition1'] + fullDF.loc[:,'Condition2']
    
    print(indCorr(fullDF, 'CondSum'))
    '''
    '''
    Despite classifying the values, we did not find a correlation even close to +0.2 with SalePrice
    Additionally, combining the two is still below a +0.2 correlation with SalePrice. We are going to drop both of these.
    '''
    
    trainInput.drop('Condition1', axis = 1)
    testInput.drop('Condition1', axis = 1)
    trainInput.drop('Condition2', axis = 1)
    testInput.drop('Condition2', axis = 1)
    
    
    '''
    Alley had many NA values, we converted them to None and checked for correlation with SalePrice. We got a high figure greater than +0.5, so 
    we are keeping it.
    '''
    '''
    fullDF.loc[:,'Alley'].fillna('None', inplace = True)
    fullDF.loc[:,'Alley'] = fullDF.loc[:,'Alley'].map(lambda val: 0 if(val=='None') else val)
    fullDF.loc[:,'Alley'] = fullDF.loc[:,'Alley'].map(lambda val: 4 if(val=='Pave') else val)
    fullDF.loc[:,'Alley'] = fullDF.loc[:,'Alley'].map(lambda val: 4 if(val== 'Grvl') else val)
    
    print(indCorr(fullDF, 'Alley'))
    
    the correlation is too low, thus we are dropping it
    '''
    testInput.drop('Alley', axis=1)
    trainInput.drop('Alley', axis=1)
    
    
    '''
    All three of: LandCountour, LotConfig, LandSlope have poor correlation with SalePrice, thus we are dropping them.
    '''
    
    testInput.drop('LandContour', axis=1)
    trainInput.drop('LandContour', axis=1)
    
    testInput.drop('LotConfig', axis=1)
    trainInput.drop('LotConfig', axis=1)
    
    testInput.drop('LandSlope', axis=1)
    trainInput.drop('LandSlope', axis=1)
    
    
    '''
    HouseStyle
    In our classification, we are coupling unfinished floors, finished floors, and the SFoyer/SLvl variables.
    
  
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='1Story') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='1.5Fin') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='1.5Unf') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='2Story') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='2.5Fin') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='2.5Unf') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SFoyer') else val)
    fullDF.loc[:,'HouseStyle'] = fullDF.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SLvl') else val)
    
    print(indCorr(fullDF, 'HouseStyle'))
    
    the correl was pretty high after classification, thus we are applying it to test and train input.
    '''
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='1Story') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='1.5Fin') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='1.5Unf') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='2Story') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='2.5Fin') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='2.5Unf') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SFoyer') else val)
    testInput.loc[:,'HouseStyle'] = testInput.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SLvl') else val)
    
    
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='1Story') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='1.5Fin') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='1.5Unf') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 6 if(val=='2Story') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 3 if(val=='2.5Fin') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 4 if(val=='2.5Unf') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SFoyer') else val)
    trainInput.loc[:,'HouseStyle'] = trainInput.loc[:,'HouseStyle'].map(lambda val: 1 if(val=='SLvl') else val)
    
    
    '''
    RoofStyle
    
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 3 if(val=='Flat') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 2 if(val=='Gable') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 0 if(val=='Gambrel') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 2 if(val=='Hip') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 2 if(val=='Mansard') else val)
    fullDF.loc[:,'RoofStyle'] = fullDF.loc[:,'RoofStyle'].map(lambda val: 0 if(val=='Shed') else val)
    
    
    print(indCorr(fullDF, 'RoofStyle'))
    
    The correlation does not meet our threshold so we are dropping it.
    '''
    
    testInput.drop('RoofStyle', axis=1)
    trainInput.drop('RoofStyle', axis=1)
    
    '''
    ExterQual & ExterCond

    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 5 if(val=='Ex') else val)
    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 4 if(val=='Gd') else val)
    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 3 if(val=='TA') else val)
    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 2 if(val=='Fa') else val)
    fullDF.loc[:,'ExterQual'] = fullDF.loc[:,'ExterQual'].map(lambda val: 1 if(val=='Po') else val)

    
    
    print(indCorr(fullDF, 'ExterQual'))
    
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 5 if(val=='Ex') else val)
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 4 if(val=='Gd') else val)
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 3 if(val=='TA') else val)
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 2 if(val=='Fa') else val)
    fullDF.loc[:,'ExterCond'] = fullDF.loc[:,'ExterCond'].map(lambda val: 1 if(val=='Po') else val)

    
    
    print(indCorr(fullDF, 'ExterCond'))
    
    There is a significant correlation of 0.6826 between ExterQual and Sale Price, thus we will keep it.
    However, there is a week correlation between ExterCond and SalePrice, thus we will not consider it as a predictor.
    '''
    
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 5 if(val=='Ex') else val)
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 4 if(val=='Gd') else val)
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 3 if(val=='TA') else val)
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 2 if(val=='Fa') else val)
    trainInput.loc[:,'ExterQual'] = trainInput.loc[:,'ExterQual'].map(lambda val: 1 if(val=='Po') else val)
    
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 5 if(val=='Ex') else val)
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 4 if(val=='Gd') else val)
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 3 if(val=='TA') else val)
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 2 if(val=='Fa') else val)
    testInput.loc[:,'ExterQual'] = testInput.loc[:,'ExterQual'].map(lambda val: 1 if(val=='Po') else val)
    

    
    '''
    Heating QC
    
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 5 if(val=='Ex') else val)
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 4 if(val=='Gd') else val)
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 3 if(val=='TA') else val)
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 2 if(val=='Fa') else val)
    fullDF.loc[:,'HeatingQC'] = fullDF.loc[:,'HeatingQC'].map(lambda val: 1 if(val=='Po') else val)
    
    print(indCorr(fullDF, 'HeatingQC'))
    
    There is a high correlation of 0.8824, thus we will do the same classification for train and testing data.
    '''
    
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 7 if(val=='Ex') else val)
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 6 if(val=='Gd') else val)
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 3 if(val=='TA') else val)
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 2 if(val=='Fa') else val)
    trainInput.loc[:,'HeatingQC'] = trainInput.loc[:,'HeatingQC'].map(lambda val: 1 if(val=='Po') else val)
    
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 7 if(val=='Ex') else val)
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 6 if(val=='Gd') else val)
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 3 if(val=='TA') else val)
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 2 if(val=='Fa') else val)
    testInput.loc[:,'HeatingQC'] = testInput.loc[:,'HeatingQC'].map(lambda val: 1 if(val=='Po') else val)
    
    
    '''
    MiscFeature
    
    In classifying MiscFeature data, we noticed that the Misc Features are few and specialized. Thus we grouped them together and compared with those
    that had none.
    fullDF.loc[:,'MiscFeature'].fillna(0, inplace=True)
    
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Elev') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Gar2') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Othr') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Shed') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='TenC') else val)
    
    print(indCorr(fullDF, 'MiscFeature'))
    
    We still got a very low correlation thus we are dropping it.
    
    
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 5 if(val=='Elev') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 4 if(val=='Gar2') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 3 if(val=='Othr') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 2 if(val=='Shed') else val)
    fullDF.loc[:,'MiscFeature'] = fullDF.loc[:,'MiscFeature'].map(lambda val: 1 if(val=='TenC') else val)
    
    data = pd.concat([fullDF.loc[:, 'SalePrice'], fullDF.loc[:,'MiscFeature']], axis=1)
    data.plot.scatter(x='MiscFeature', y='SalePrice', ylim=(0,800000));
    
    testInput.drop('MiscFeature', axis=1)
    trainInput.drop('MiscFeature', axis=1)
    '''
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
    



    
# ===============================================================================
def standardize(inputDF, cols):
    inputDF.loc[:, cols] = (inputDF.loc[:, cols] - inputDF.loc[:, cols].mean()) /inputDF.loc[:, cols].std()
    return inputDF.loc[:, cols]
    

def indCorr(df, colName):
    return df.loc[:, colName].corr(df.loc[:, 'SalePrice'])


'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

