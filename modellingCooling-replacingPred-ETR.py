# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:34:57 2024

@author: iaavenda
"""
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from quantile_forest import ExtraTreesQuantileRegressor

from sklearn.linear_model import LinearRegression
# from lineartree import LinearForestRegressor

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


import os, sys
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

file_path_package = os.path.join('linearExtraForest', 'linearextraforest.py')
sys.path.append(os.path.abspath(file_path_package))

# from os.path.abspath(os.path.join('linearExtraForest', 'linearextraforest.py')) import LinearForestRegressor
from linearExtraForest import LinearExtraForestRegressor

lags_order = 4
window = 1*24*7*5


folder_name = 'citylearn_challenge_2023_phase_3_3'
name_file = 'Building_1.csv'
file_path = os.path.join(folder_name, name_file)
df = pd.read_csv(file_path,header=0,index_col=0) #,parse_dates=True
df.loc[df.loc[:,'hour'] == 24,'hour'] = 0
df.index = '2024-' + df.index.astype('str') + '-' + df.loc[:,'day_type'].astype('str') + ' ' + df.loc[:,'hour'].astype('str') + ':00:00'

target_name = 'cooling_demand'
setpoint_name = 'indoor_dry_bulb_temperature_set_point'

df= df.loc[:,[target_name,setpoint_name,'indoor_dry_bulb_temperature']]

# df= df.loc[:,[target_name,setpoint_name]]

#%%

name_weather = 'weather.csv'
weather_path = os.path.join(folder_name, name_weather)
weather_df = pd.read_csv(weather_path,header=0) #,parse_dates=True
weather_df.index = df.index
weather_df = weather_df.loc[:,['outdoor_dry_bulb_temperature','direct_solar_irradiance']]
# weather_df = weather_df.loc[:,['outdoor_dry_bulb_temperature',	'outdoor_relative_humidity', 'diffuse_solar_irradiance', 'direct_solar_irradiance']]
#%%

df = pd.concat([df,weather_df],axis=1)
df.index = pd.to_datetime(df.index)
df.index = pd.date_range(str(df.index[0]), periods=df.shape[0], freq='H')

df.loc[:,setpoint_name] = (df.loc[:,'indoor_dry_bulb_temperature'].shift(1) - df.loc[:,setpoint_name]).round(2)
# df.loc[:,'diffTemp'] = (df.loc[:,'indoor_dry_bulb_temperature'] - df.loc[:,'outdoor_dry_bulb_temperature']).round(2)
# df.loc[df.loc[:,setpoint_name] <= 0,setpoint_name] = 0
df.drop(columns=['indoor_dry_bulb_temperature'])
df = df.iloc[1:,:]

# df.loc[:,target_name] = df.loc[:,target_name] - df.loc[:,target_name].shift(1)
# df = df.iloc[1:,:]

target_train = df.loc[:,[target_name]]
exg = df.drop(columns=[target_name])

#%% FOR ML MODELS

for j in range(1,lags_order+1):
    for col in df.columns:
        exg.loc[:,col+'-'+str(j)+'lag'] = df[col].shift(j)
        # exg.loc[:,col+'-'+str(j)+'diff'] = df[col].shift(j+1) - df[col].shift(j)
        # exg.loc[:,col+'-'+str(j)+'MA'] = (df[col].shift(j+1) + df[col].shift(j))/2

keep_index = exg.filter(like='lag').columns

exg['hour_sin'] = np.sin(2 * np.pi * (exg.index.hour)/24.0)
exg['day_sin'] = np.sin(2 * np.pi * exg.index.dayofweek/7.0)

exg['hour_cos'] = np.cos(2 * np.pi * (exg.index.hour)/24.0)
exg['day_cos'] = np.cos(2 * np.pi * exg.index.dayofweek/7.0)


exg.dropna(inplace=True)
# exg.drop(columns=['indoor_dry_bulb_temperature','indoor_relative_humidity','outdoor_dry_bulb_temperature',	'outdoor_relative_humidity', 'diffuse_solar_irradiance', 'direct_solar_irradiance'])
# exg.drop(columns=['indoor_dry_bulb_temperature','outdoor_dry_bulb_temperature','direct_solar_irradiance','diffTemp'])
exg.drop(columns=['outdoor_dry_bulb_temperature','direct_solar_irradiance'])

target_train = target_train.loc[exg.index,:]

#%%

# lenCV = round((len(target_train)-window)/(folds+1))

n_estimators= 70

results_DF_val = pd.DataFrame([])
for k in range(window,len(target_train),1):
# for k in range(window,len(target_train),lenCV):
    
    target_val = target_train.iloc[k-window:k] #[train_len*k-window:train_len*k]
    exg_val = exg.iloc[k-window:k] #[train_len*k-window:train_len*k,:]
    
    target_test = target_train.iloc[k]
    exg_test = exg.iloc[[k],:]
    
    # model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    # model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    # model = XGBRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    # model = XGBRegressor(n_estimators=n_estimators, booster='gblinear', random_state=42, n_jobs=-1)
    
    # model = ExtraTreesQuantileRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    # model = LinearForestRegressor(base_estimator=LinearRegression(), random_state=42, n_jobs=-1)
    model = LinearExtraForestRegressor(base_estimator=LinearRegression(), random_state=42, n_jobs=-1)
    
    ML_CV = model.fit(exg_val, target_val)
    target_hat = ML_CV.predict(exg_test)
    
    results_DF_val.loc[target_test.name,'target_real'] = target_test.item() 
    results_DF_val.loc[target_test.name,'target_pred'] = target_hat
    
    # replacing future value with the forecasting
    target_train.iloc[k] = target_hat
    
print(r2_score(results_DF_val.loc[:,'target_real'],results_DF_val.loc[:,'target_pred']))
print(mean_absolute_percentage_error(results_DF_val.loc[results_DF_val.iloc[:,0] > 0,:].loc[:,'target_real'],results_DF_val.loc[results_DF_val.iloc[:,0] > 0,:].loc[:,'target_pred']))

results_DF_val.plot()
