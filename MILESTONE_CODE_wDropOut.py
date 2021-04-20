import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate,cross_val_score
import torch

InputDataDir='data/GOSHIP_Data/QCFilteredData.csv'
FigDir = 'figures/LinearRegression_wDropOut/'

pos_encode=['raw','radians']
date_encode=['thermometer','sincos']

P_cv_scores = np.zeros((len(pos_encode), len(date_encode)))
P_cv_scores[:]=np.NaN

P_cv_Rscores = np.zeros((len(pos_encode), len(date_encode)))
P_cv_Rscores[:]=np.NaN

P_cv_var = np.zeros((len(pos_encode), len(date_encode)))
P_cv_var[:]=np.NaN

S_cv_scores = np.zeros((len(pos_encode), len(date_encode)))
S_cv_scores[:]=np.NaN

S_cv_Rscores = np.zeros((len(pos_encode), len(date_encode)))
S_cv_Rscores[:]=np.NaN

S_cv_var = np.zeros((len(pos_encode), len(date_encode)))
S_cv_var[:]=np.NaN

P_models=[[]]*(len(pos_encode)*len(date_encode))
S_models=[[]]*(len(pos_encode)*len(date_encode))

train_val = int(np.floor(0.9*pd.read_csv(InputDataDir).shape[0]))
print('Number training points: ',train_val)
print('Total # Data points: ', pd.read_csv(InputDataDir).shape[0])
k=10

z=0
def mean_square_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = mean_squared_error(y, y_pred)
    return cm

def BernMask(ind, p):
    import torch
    init_p = torch.empty(len(ind),1)
    init_p[:] = p
    bmask = torch.bernoulli(init_p)
    bmask=np.array(bmask)
    #print(bmask)
    
    return bmask

p = 0.7

i = 0 
j = 0
# Load data
GOSHIP_Data=pd.read_csv(InputDataDir)
GOSHIP_Data=GOSHIP_Data.iloc[:,3:]

# Shuffle data
GOSHIP_Data = GOSHIP_Data.sample(frac=1)

Position_Data = GOSHIP_Data.loc[:,['LATITUDE','LONGITUDE']]
Date_Data = GOSHIP_Data.loc[:,'MONTH']

# Standardize non-position/datae data
scaler = StandardScaler().fit(GOSHIP_Data.iloc[:,2:-1])
data_scaled = scaler.transform(GOSHIP_Data.iloc[:,2:-1])
data_scaled = pd.DataFrame(data_scaled, columns = ['PRES','TEMP','SAL','OXY','NITR','PHSP','SILI'])

# Get X data
X=data_scaled.iloc[:,0:5] # Pres, temp, sal , oxy, nitr

# Encode position data and add it to X-data
if pos_encode[i]=='none':
    # Do not include postion data
    pass
elif pos_encode[i]=='raw':
    # Use raw lat/lon data
    X['LAT']=GOSHIP_Data.loc[:,'LATITUDE'].to_numpy()
    X['LON']=GOSHIP_Data.loc[:,'LONGITUDE'].to_numpy()
elif pos_encode[i]=='radians':
    # Use lat/lon encoded as radians
    X['LAT']=np.radians(GOSHIP_Data.loc[:,'LATITUDE'].to_numpy())
    X['LON']=np.radians(GOSHIP_Data.loc[:,'LONGITUDE'].to_numpy())

# Encode date data and add it to X-data
if date_encode[i]== 'none':
    pass
elif date_encode[i]== 'thermometer':
    non_data=GOSHIP_Data.loc[:,'MONTH'].to_numpy()
    temp_data=np.zeros((len(non_data), 12))
    for m in np.arange(len(temp_data)):
        m_ind=non_data[m]  
        temp_data[m,:m_ind]=1

    EncodedFeatures=pd.DataFrame(temp_data, columns=['M1','M2','M3', 'M4', 'M5','M6', 'M7', 'M8', 'M9','M10','M11', 'M12'])
    #print(EncodedFeatures)
    X=pd.concat([X,EncodedFeatures], axis=1)
    #print(X)
elif date_encode[i]== 'sincos':
    # Encode as a sin/cosine pair
    X['MONTH_SIN']=np.sin((2*np.pi*GOSHIP_Data.loc[:,'MONTH'])/max(GOSHIP_Data.loc[:,'MONTH']))
    X['MONTH_COS']=np.cos((2*np.pi*GOSHIP_Data.loc[:,'MONTH'])/max(GOSHIP_Data.loc[:,'MONTH']))

# Get Y data
Y_P = data_scaled.loc[:,'PHSP']
Y_S = data_scaled.loc[:, 'SILI']

# Run regression
print('\nMonth encoding: ', pos_encode[i])
print('Date encoding: ', date_encode[j])

# Split in to test and train values
X_train = X.iloc[:train_val,:]
X_test=X.iloc[train_val:, :]

Y_P_train=Y_P.iloc[:train_val]
Y_P_test=Y_P.iloc[train_val:]

Y_S_train=Y_S.iloc[:train_val]
Y_S_test=Y_S.iloc[train_val:]

ColVal=X.columns.to_list()
IndexList=X_train.index.to_list()

# For dropout, apply bernoulli mask to training data
init_p = torch.empty(X_train.shape)
init_p[:] = p
bmask_train_array = torch.bernoulli(init_p)
bmask_train_array=np.array(bmask_train_array)

bmask_train_df=pd.DataFrame(bmask_train_array, index=IndexList, columns=ColVal)
X_train=X_train*bmask_train_df
# Print Linear regression with cross validation 
linear=LinearRegression()
cv_results_P = cross_validate(linear, X_train, Y_P_train, cv=k,return_train_score=True, scoring=mean_square_scorer, return_estimator=True)
cv_results_S = cross_validate(linear, X_train, Y_S_train, cv=k,return_train_score=True,scoring=mean_square_scorer, return_estimator=True)

print('\nCross validation with k-folding: ', k)
print('\n%% PHOSPHATE %%')
print('train score: ', cv_results_P['train_score'])
print('test score: ', cv_results_P['test_score'])

# Select the one with the lowest scores
print('Estimator: ', cv_results_P['estimator'][np.argmin(cv_results_P['test_score'])].coef_)
best_P_model = cv_results_P['estimator'][np.argmin(cv_results_P['test_score'])]
print('R2 values: ',best_P_model.score(X_train,Y_P_train))
P_models[z]=best_P_model
# Test this model and calculate MSE on test data
P_pred_test_cv=best_P_model.predict(X_test)
P_cv_Rscores[i,j]=best_P_model.score(X_train,Y_P_train)
P_cv_scores[i,j]=mean_squared_error(Y_P_test, P_pred_test_cv)
print('testing error: ', mean_squared_error(Y_P_test, P_pred_test_cv))

## Droup out testing
# Calculate E
f_i_P=np.zeros(Y_P_test.shape[0])
f_i_P[:]=np.NaN
for k in np.arange(Y_P_test.shape[0]):
    bmask = BernMask(ind=X.columns.to_list(), p=p)

    old_coef_P=best_P_model.coef_.reshape((best_P_model.coef_.shape[0],1))
    #print(old_coef_P*bmask)
    new_coef_P = old_coef_P*bmask
    const_P =best_P_model.intercept_
    #print('New coef: ', new_coef_P)

    # Evaluate new value of P with masked coeff
    x_i = X_test.iloc[k,:].to_numpy()
    x_i = x_i.reshape((x_i.shape[0],1))
    f_i_P[k]=np.dot(new_coef_P.T, x_i)[0,0]+const_P

E_P=np.nanmean(f_i_P)
var_P = np.nanmean((P_pred_test_cv**2 - E_P**2))
P_cv_var[i,j]=var_P

print('Dropout E: ', E_P)
print('Dropout Var: ', var_P)

print('\n%% SILICATE %%')

print('train score: ', cv_results_S['train_score'])
print('test score: ', cv_results_S['test_score'])
print('Estimator: ', cv_results_S['estimator'][np.argmin(cv_results_S['test_score'])].coef_)
best_S_model = cv_results_S['estimator'][np.argmin(cv_results_S['test_score'])]
print('R2 values: ',best_S_model.score(X_train,Y_S_train))
S_pred_test_cv=best_S_model.predict(X_test)
S_cv_Rscores[i,j]=best_S_model.score(X_train,Y_S_train)
S_cv_scores[i,j]=mean_squared_error(Y_S_test, S_pred_test_cv)
print('testing error: ', mean_squared_error(Y_S_test, S_pred_test_cv))
S_models[z]=best_S_model

f_i_S=np.zeros(Y_S_test.shape[0])
f_i_S[:]=np.NaN
for k in np.arange(Y_S_test.shape[0]):
    bmask = BernMask(ind=X.columns.to_list(), p=p)

    old_coef_S=best_S_model.coef_.reshape((best_S_model.coef_.shape[0],1))
    #print(old_coef_P*bmask)
    new_coef_S = old_coef_S*bmask
    const_S =best_S_model.intercept_
    #print('New coef: ', new_coef_P)

    # Evaluate new value of P with masked coeff
    x_i = X_test.iloc[k,:].to_numpy()
    x_i = x_i.reshape((x_i.shape[0],1))
    f_i_S[k]=np.dot(new_coef_S.T, x_i)[0,0]+const_S

E_S=np.nanmean(f_i_S)
var_S = np.nanmean((S_pred_test_cv**2 - E_S**2))
S_cv_var[i,j]=var_S
print('Dropout E: ', E_S)
print('Dropout Var: ', var_S)

# # Plot results
# fig, axs = plt.subplots(1,2, figsize=(8,6))

# x_P = np.arange(np.nanmin([Y_P_test, P_pred_test_cv]),np.nanmax([Y_P_test, P_pred_test_cv])+1)
# x_S = np.arange(np.nanmin([Y_S_test, S_pred_test_cv]),np.nanmax([Y_S_test, S_pred_test_cv])+1)

# ax = plt.subplot(1,2,1)
# ax.scatter(Y_P_test, P_pred_test_cv)
# ax.plot(x_P, x_P, 'k-')
# ax.set_xlabel('P test (std)')
# ax.set_ylabel('P predicted (std)')
# ax.set_title('Phosphate')

# ax = plt.subplot(1,2,2)
# ax.scatter(Y_S_test, S_pred_test_cv)
# ax.plot(x_S, x_S, 'k-')
# ax.set_xlabel('S test (std)')
# ax.set_ylabel('S predicted (std)')
# ax.set_title('Silicate')

# fig.suptitle('Test Results for: '+pos_encode[i]+' & '+date_encode[j])
# plt.subplots_adjust(wspace= .5)

# plt.savefig(FigDir+'CV_PosEncode_'+pos_encode[i]+'_DateEncode_'+date_encode[j]+'.jpg')

