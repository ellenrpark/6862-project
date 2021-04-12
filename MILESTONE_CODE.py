#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate,cross_val_score


# In[83]:


InputDataDir='data/GOSHIP_Data/QCFilteredData.csv'
FigDir = 'figures/LinearRegression/'

pos_encode=['none','raw','radians']
date_encode=['none','thermometer','sincos']

P_cv_scores = np.zeros((len(pos_encode), len(date_encode)))
P_cv_scores[:]=np.NaN

S_cv_scores = np.zeros((len(pos_encode), len(date_encode)))
S_cv_scores[:]=np.NaN

P_models=[[]]*(len(pos_encode)*len(date_encode))
S_models=[[]]*(len(pos_encode)*len(date_encode))


# In[88]:


train_val = int(np.floor(0.9*pd.read_csv(InputDataDir).shape[0]))
print(train_val)
k=10

z=0
def mean_square_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = mean_squared_error(y, y_pred)
    return cm

for i in np.arange(len(pos_encode)):
    for j in np.arange(len(date_encode)):
        
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
        
#         # Run Linear regression
#         print('\nLinear Regression with training size: ', train_val,' and testing size :', len(Y_S)-train_val)
#         # Phosphate
#         P_reg=LinearRegression().fit(X_train,Y_P_train)
#         P_pred_train=P_reg.predict(X_train)
#         P_pred_test=P_reg.predict(X_test)
        
#         print('\n%% PHOSPHATE %%')
#         print("score: ", P_reg.score(X_train,Y_P_train))
#         print("coefficients: ", P_reg.coef_)
#         print("intercept: ", P_reg.intercept_)
#         print('training error: ', mean_squared_error(Y_P_train, P_pred_train))
#         print('testing error: ', mean_squared_error(Y_P_test, P_pred_test))
        
#         # Silicate
#         S_reg=LinearRegression().fit(X_train,Y_S_train)
#         S_pred_train=S_reg.predict(X_train)
#         S_pred_test=S_reg.predict(X_test)
        
#         print('\n%% SILICATE %%')
#         print("score: ", S_reg.score(X_train,Y_S_train))
#         print("coefficients: ", S_reg.coef_)
#         print("intercept: ", S_reg.intercept_)
#         print('training error: ', mean_squared_error(Y_S_train, S_pred_train))
#         print('testing error: ', mean_squared_error(Y_S_test, S_pred_test))
        
        # Print Linear regression with cross validation 
        linear=LinearRegression()
        cv_results_P = cross_validate(linear, X, Y_P, cv=k,return_train_score=True, scoring=mean_square_scorer, return_estimator=True)
        cv_results_S = cross_validate(linear, X, Y_S, cv=k,return_train_score=True,scoring=mean_square_scorer, return_estimator=True)
        
        print('\nCross validation with k-folding: ', k)
        print('\n%% PHOSPHATE %%')
        print('train score: ', cv_results_P['train_score'])
        print('test score: ', cv_results_P['test_score'])
        
        # Select the one with the lowest scores
        print('Estimator: ', cv_results_P['estimator'][np.argmin(cv_results_P['test_score'])].coef_)
        
        best_P_model = cv_results_P['estimator'][np.argmin(cv_results_P['test_score'])]
        P_models[z]=best_P_model
        # Test this model and calculate MSE on test data
        P_pred_test_cv=best_P_model.predict(X_test)
        P_cv_scores[i,j]=mean_squared_error(Y_P_test, P_pred_test_cv)
        print('testing error: ', mean_squared_error(Y_P_test, P_pred_test_cv))
        
        print('\n%% SILICATE %%')
        print('train score: ', cv_results_S['train_score'])
        print('test score: ', cv_results_S['test_score'])
        print('Estimator: ', cv_results_S['estimator'][np.argmin(cv_results_S['test_score'])].coef_)
        best_S_model = cv_results_S['estimator'][np.argmin(cv_results_S['test_score'])]
        S_pred_test_cv=best_S_model.predict(X_test)
        S_cv_scores[i,j]=mean_squared_error(Y_S_test, S_pred_test_cv)
        print('testing error: ', mean_squared_error(Y_S_test, S_pred_test_cv))
        S_models[z]=best_S_model
        
        # Plot results
        
        fig, axs = plt.subplots(1,2, figsize=(8,6))
        
        x_P = np.arange(np.nanmin([Y_P_test, P_pred_test_cv]),np.nanmax([Y_P_test, P_pred_test_cv]))
        x_S = np.arange(np.nanmin([Y_S_test, S_pred_test_cv]),np.nanmax([Y_S_test, S_pred_test_cv]))
        ax = plt.subplot(1,2,1)
        ax.scatter(Y_P_test, P_pred_test_cv)
        ax.plot(x_P, x_P, 'k-')
        ax.set_xlabel('P test (std)')
        ax.set_ylabel('P predicted (std)')
        ax.set_title('Phosphate')
        
        ax = plt.subplot(1,2,2)
        ax.scatter(Y_S_test, S_pred_test_cv)
        ax.plot(x_S, x_S, 'k-')
        ax.set_xlabel('S test (std)')
        ax.set_ylabel('S predicted (std)')
        ax.set_title('Silicate')
        
        fig.suptitle('Test Results for: '+pos_encode[i]+' & '+date_encode[j])
        plt.subplots_adjust(wspace= .5)

        plt.savefig(FigDir+'CV_PosEncode_'+pos_encode[i]+'_DateEncode_'+date_encode[j]+'.jpg')
                    
        z = z+1
print('\n%% FINAL RESULTS %%')
P_cv_scores=pd.DataFrame(P_cv_scores, index=pos_encode, columns=date_encode)
print('\nPhosphate')
P_cv_scores
    
print('\nSilicate')
S_cv_scores=pd.DataFrame(S_cv_scores, index=pos_encode, columns=date_encode)
S_cv_scores
        
  
            


# In[78]:


P_cv_scores


# In[79]:


S_cv_scores


# In[87]:





# In[ ]:




