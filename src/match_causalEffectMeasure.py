import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import cdist


### Processing variable types
cellData = pd.read_csv('../cell2celltrain.csv')
def checkCountDistinct(df,catVars,threshold):
    catList=list()
    for cat in catVars:
        disCount=len(df[cat].value_counts())
        
        if disCount<threshold:
            catList.append(cat)
        else:
            pass
    
    return catList

predictors = [col for col in cellData.columns if col not in ['CustomerID','Churn','RetentionCalls','RetentionOffersAccepted']]

catVarLst1 = checkCountDistinct(cellData,predictors,threshold=10)
catVarLst2 = ['UniqueSubs','ActiveSubs','ServiceArea','Handsets','HandsetModels','ReferralsMadeBySubscriber','IncomeGroup',
              'AdjustmentsToCreditRating','HandsetPrice']
catVarLst = catVarLst1 + catVarLst2
numVarLst = [col for col in predictors if col not in catVarLst]

#### Setting variable types for xgboost model
lbl = preprocessing.LabelEncoder()
for cat in catVarLst:
    cellData[cat] = lbl.fit_transform(cellData[cat].astype(str))

cellData[numVarLst] = cellData[numVarLst].astype(float)

### Processing treatment variable for propensity model
cellData['RetentionCallsAction'] =  np.where(cellData['RetentionCalls']>0,1,0)

### Building propensity score model
#####***we have have one pair of treatment pairs, giving retention call and not, which will be used a target variable to build the propensity score model***#####

xgbPropensity = xgb.XGBClassifier()
xgbPropensity.fit(cellData[predictors],cellData['RetentionCallsAction'])

propensityScore = xgbPropensity.predict_proba(cellData[predictors])[:,1]
cellData.loc[:,'propensityToReceiveCall'] = propensityScore

#####***Calculate average standardized absolute mean difference in the covariates (ASAM): The goal of the propensity score model is to minimize the ASAM. For each covariate, we calculated the absolute value of the difference between the mean for the treatment group and the weighted mean for the comparison group, divided by the standard deviation for the treatment group. We subsequently averaged these values across all covariates to obtain the ASAM. ***#####

cellDataTreatment = cellData[cellData['RetentionCallsAction']==1].reset_index(drop=True)
cellDataComparison = cellData[cellData['RetentionCallsAction']==0].reset_index(drop=True)

sumMeanDiff = 0
for convariate in predictors:
    
    # The weights for subject l in the comparison group are defined by the odds that a subject with 
    # covariates x_l will be assigned to treatment.
    
    weight = cellDataComparison['RetentionCallsAction']/(1-cellDataComparison['RetentionCallsAction'])
    mean_diff = np.abs(np.mean(cellDataTreatment[convariate]) - np.mean(cellDataComparison[convariate]*weight))
    std_dev = np.std(cellDataTreatment[convariate])
    
    if std_dev> 0:
        sumMeanDiff = sumMeanDiff + (mean_diff/std_dev)
    else:
        sumMeanDiff = sumMeanDiff
    
asam = mean_diff / len(predictors)

print('The average standardized absolute mean difference in the covariates (ASAM) is: ')
print(asam)

### Matching
cellDataTreatment = cellDataTreatment.fillna(0) #users need to adapt imputation method according to use case
cellDataComparison = cellDataComparison.fillna(0)

featuresForDistance = predictors + ['propensityToReceiveCall']

# Calculate the Mahalanobis distance matrix
distances = cdist(cellDataTreatment[featuresForDistance], cellDataComparison[featuresForDistance], 'mahalanobis')

propensityCaliper = 0.1 #users need to adapt this value according to use case
nearest_neighbors = []
for i in range(len(cellDataTreatment)):
    indices = np.argsort(distances[i])[:1]
    filtered_indices = []
    for j in indices:
        if abs(cellDataTreatment.iloc[i].propensityToReceiveCall - cellDataComparison.iloc[j].propensityToReceiveCall) < propensityCaliper:
            filtered_indices.append(j)
    nearest_neighbors.append(cellDataComparison.iloc[filtered_indices])

matched_ids = []
for i in range(len(nearest_neighbors)):
    treatment_id = cellDataTreatment.loc[i, 'CustomerID']
    comparison_indices = nearest_neighbors[i].index.tolist()
    comparison_ids = cellDataComparison.loc[comparison_indices, 'CustomerID'].tolist()
    matched_id = comparison_ids[0] if comparison_ids else None
    matched_ids.append(matched_id)

cellDataTreatment['matchedCustomerID_comparison'] = matched_ids


### Causal effect measurement

#### Selected matched pairs with CustomerID from treatment group, and matchedCustomerID_comparison from comparison group
cellDataTreatmentIDs = cellDataTreatment[['CustomerID','matchedCustomerID_comparison']]

#### Read churn probabiltiy from global lapse model 
churnProbWithCalls = pd.read_csv('../cellWithCalls_ChurnProbability.csv')
churnProbWithoutCalls = pd.read_csv('../cellWithoutCalls_ChurnProbability.csv')

#### Got churn probabilties for treatment and comparison customers within the matched pairs
cellDataTreatmentIDs = pd.merge(cellDataTreatmentIDs,churnProbWithCalls[['CustomerID','churnProbability']],on='CustomerID')
cellDataTreatmentIDs = pd.merge(cellDataTreatmentIDs,churnProbWithoutCalls[['CustomerID','churnProbability']],left_on='matchedCustomerID_comparison',right_on='CustomerID')

#### Computed average treatment effect on customers' lapse/retention
cellDataTreatmentIDs = cellDataTreatmentIDs.drop('CustomerID_y',axis=1)
cellDataTreatmentIDs.columns = ['CustomerID','matchedCustomerID_comparison','churnProbabilityWithCall','churnProbabilityWithoutCall']
average_treat_causal_effect = (cellDataTreatmentIDs['churnProbabilityWithCall'] -cellDataTreatmentIDs['churnProbabilityWithoutCall']).mean()

print("The average treatment effect  is: ")
print(average_treat_causal_effect)
