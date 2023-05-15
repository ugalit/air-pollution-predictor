'''
Final Project of Data Science Bootcamp at Spiced Academy
Silke R. Schmidt, December 2022
Project: prediction of particulate matter in Berlin using traffic and weather data
--- READ, PROCESS AND VISUALIZE DATA---
'''

import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# function definitions
def read_air(filename):
    '''read Berlin air quality data'''
    df = pd.read_csv(filename, sep=';', header=0, skiprows=[1,2,3], skip_blank_lines=True)
    df.rename(columns={'Station':'datetime'}, inplace=True)
    df.datetime = pd.to_datetime(df.datetime, format='%d.%m.%Y %H:%M')
    param = filename.split('_')[1]
    colname = pd.read_csv(filename, sep=';', nrows=1).iloc[0][1]
    globals()[param] = pd.melt(df, id_vars='datetime', var_name='station', value_name=colname)
    return param, colname

def read_traffic(path, filename):
    '''read Berlin traffic data'''
    df = pd.read_csv(path+filename, sep=';', header=0)
    df['datetime'] = pd.to_datetime(df.tag + df.stunde.astype(str), format='%d.%m.%Y%H')
    df.drop(columns=['tag', 'stunde', 'qualitaet'], inplace=True)
    param = filename.split('.')[0]
    globals()[param] = df
    return param

def read_weather(fname):
    '''read DWD weather data from Berlin'''
    dat = pd.read_csv(fname, sep=';', header=0, skipinitialspace=True, na_values=-999)
    dat['datetime'] = pd.to_datetime(dat.MESS_DATUM, format='%Y%m%d%H')
    dat.drop(columns=dat.columns[-2], inplace=True)
    dat.drop(columns=['STATIONS_ID', 'MESS_DATUM', 'QUALITAETS_NIVEAU', 'STRUKTUR_VERSION'], inplace=True)
    param = fname.split('/')[3].split('_')
    if len(param)==5:
        param = param[1]+param[2]
    else:
        param = param[1]
    globals()[param] = dat
    return param

def plot_data(df, y, hue, legend, plotpath):
    '''plot time series data'''
    sns.set(rc={"figure.figsize":(9, 5)})
    _ = plt.figure()
    if hue is not None:
        img = sns.lineplot(data=df, x='datetime', y=y, hue=hue, alpha=0.5, legend=legend)
    else:
        img = sns.lineplot(data=df, x='datetime', y=y, alpha=0.5, legend=legend)
    if legend:
        sns.move_legend(img, "center left", bbox_to_anchor=(1.02, 0.55))
    fig = img.get_figure()
    fig.savefig('plots/' + plotpath + y + '.png', bbox_inches="tight")
    fig.clf()


#
# read, plot and clean data
#

# air quality

params = []
colnames = []
files = glob.glob('data/Berlin/air/*.csv')
print('reading air quality data')
for filename in files:
    params.append(read_air(filename)[0])
    colnames.append(read_air(filename)[1])

for df,y in zip(params,colnames):
   print(df, y)
   plot_data(eval(df), y, 'station', True, 'air/raw/')

pm2.loc[pm2[pm2['Feinstaub (PM2,5)']>100].index, 'Feinstaub (PM2,5)'] = np.nan
pm10.loc[78016, 'Feinstaub (PM10)'] = np.nan

pm2.to_csv('data/df_pm2.5_clean.csv')

for df,y in zip(params,colnames):
   print(df, y)
   plot_data(eval(df), y, 'station', True, 'air/clean/')

# traffic

TRAFFIC_PATH = 'data/Berlin/traffic/Messquerschnitt/'
mq_list = os.listdir(TRAFFIC_PATH)
df_traffic = []
print('reading traffic data')
for filename in mq_list:
    df_traffic.append(read_traffic(TRAFFIC_PATH, filename))

for df in df_traffic:
   print(df)
   plot_data(data=eval(df), y='q_kfz_mq_hr', hue='mq_name', legend=False, plotpath='traffic/raw/')

traffic = eval(df_traffic[0])
for df in df_traffic[1:]:
    list_df = [traffic, eval(df)]
    traffic = pd.concat(list_df)
traffic.sort_values(by=['datetime', 'mq_name'], inplace=True, ignore_index=True)

traffic.loc[traffic.v_kfz_mq_hr<0, 'v_kfz_mq_hr'] = np.nan
traffic.loc[traffic.v_pkw_mq_hr<0, 'v_pkw_mq_hr'] = np.nan
traffic.loc[traffic.v_lkw_mq_hr<0, 'v_lkw_mq_hr'] = np.nan

traffic.to_csv('data/df_traffic_clean.csv')

# weather data

WEATHER_PATH = 'data/Berlin/weather'
WEATHER_PARAMS = ['LUFTTEMPERATUR', 'REL_FEUCHTE', 'NIEDERSCHLAGSHOEHE',
    'WINDGESCHWINDIGKEIT', 'WINDRICHTUNG', 'SONNENSCHEINDAUER']
w_list = glob.glob(WEATHER_PATH+'/**/produkt*akt.txt', recursive=True)

print('reading weather data')
df_weather = []
for filename in w_list:
    df_weather.append(read_weather(filename))

airtemperature.drop(columns='STRAHLUNGSTEMPERATUR', inplace=True)
pressure.drop(columns='LUFTDRUCK_NN', inplace=True)

weather = pd.merge(airtemperature, precipitation, how='outer', on='datetime')
weather = pd.merge(weather, wind, how='outer', on='datetime')
weather = pd.merge(weather, sun, how='outer', on='datetime')
weather.sort_values(by='datetime', inplace=True)

weather = weather[weather['datetime'] >= max(min(traffic.datetime), min(pm2.datetime))]

weather.to_csv('data/df_weather_clean.csv')

for y in WEATHER_PARAMS:
   plot_data(df=weather, y=y, hue=None, legend=False, plotpath='weather/raw/')

# Merge subset for station Silbersteinstrasse

print('subsetting and merging data for subset Silberstein')
traffic_Silberstein = traffic[traffic['mq_name']=='TE385']
traffic_Silberstein = traffic_Silberstein[['datetime', 'q_kfz_mq_hr', 'v_kfz_mq_hr']].reset_index(drop=True)
pm_Silberstein = pm2[pm2['station']=='143 Silbersteinstraße']
pm_Silberstein = pm_Silberstein[['datetime', 'Feinstaub (PM2,5)']].reset_index(drop=True)
weather.reset_index(drop=True, inplace=True)

df_silber = pd.merge(traffic_Silberstein, pm_Silberstein, how='outer', on='datetime')
df_silber = pd.merge(df_silber, weather, how='outer', on='datetime')
df_silber = df_silber[df_silber['datetime']>='2021-12-08 11:00:00']
df_silber = df_silber[df_silber['datetime']<='2022-10-31 15:00:00']
df_silber.sort_values(by='datetime', inplace=True, ignore_index=True)

print('saving datasets to disk')
df_silber.to_pickle('data/df_Silberstein.pkl')
df_silber.to_csv('data/df_Silberstein.csv')
