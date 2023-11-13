import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing


### Reading data
cellData = pd.read_csv('../cell2celltrain.csv')

### Processing variable types
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

### Processing treatment groups
######*****Here used 'RetentionCalls' as a treatment indicator****#####
cellData['RetentionCallsAction'] =  np.where(cellData['RetentionCalls']>0,1,0)
cellData['ChurnEdited'] =  np.where(cellData['Churn']=='Yes',1,0)
cellDataCalls = cellData[cellData['RetentionCallsAction']==1]
cellDataNoCalls = cellData[cellData['RetentionCallsAction']==0]

### Building lapse model by treatment groups
xgbCalls = xgb.XGBClassifier()
xgbNoCalls = xgb.XGBClassifier()

xgbCalls.fit(cellDataCalls[predictors],cellDataCalls['ChurnEdited'])
xgbNoCalls.fit(cellDataNoCalls[predictors],cellDataNoCalls['ChurnEdited'])

######*****Computed each user's churn probability****#####
churnCalls = xgbCalls.predict_proba(cellDataCalls[predictors])[:,1]
churnNoCalls = xgbNoCalls.predict_proba(cellDataNoCalls[predictors])[:,1]

cellDataCalls.loc[:,'churnProbability'] = churnCalls
cellDataNoCalls.loc[:,'churnProbability'] = churnNoCalls

######*****Wrote out churn probability for causal effect measurement****#####

cellDataCalls.to_csv('../cellWithCalls_ChurnProbability.csv')
cellDataNoCalls.to_csv('../cellWithoutCalls_ChurnProbability.csv')
