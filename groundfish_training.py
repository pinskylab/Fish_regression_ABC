# this script extract data from a .csv file called jude_cod.csv. Extract the species whose dynamcs we are interested in modelling  and the variables that affects the abundance of species. One of the variable in Question is Temperature. We use the bottom teperature measurements. Instead of using the time series for temperature in jude_cod.csv, we use the ROMS temperature values, found in the .csv file haul_ROMS.csv.
from __future__ import print_function, division
import sys
import numpy as np
import random as random
import math as math
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from random import triangular
import scipy.stats as sst
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Extracting data from tehe .csv file jude_cod.csv
df = pd.read_csv("jude_cod.csv", usecols=['haulid','sppocean','NUMLEN', 'year', 'lat', 'bottemp','region','survey','lengthclass'])
#Extracting  bottom temperature values from the ROMS data
df1 = pd.read_csv("haul_ROMS.csv", usecols=['haulid',  'temp_bottom', 'temp_surface'])
#Selecting the years to use. Part of the data was used for parameter estimation  and the remaining data was used for model validation .
df=df.loc[(df['year'] <2013)]
#selecting teh species to use
df=df.loc[(df['sppocean'] =='gadus morhua_Atl')]
#selecting the season when the species was caught
df=df.loc[(df['survey'] =='NEFSC_NEUSSpring')]
#rouding up latitudes to integers
df.lat = df.lat.round().astype(np.int)
# taking care of missing data in both data frame
df=df.interpolate(method ='linear', limit_direction ='forward')
df=df.interpolate(method ='linear', limit_direction ='backward')
df1=df1.interpolate(method ='linear', limit_direction ='forward')
df1=df1.interpolate(method ='linear', limit_direction ='backward')
#Merging the the two data frame so as to match the temp values in the ROMS data to the species abundance data
df=pd.merge(df, df1, on='haulid')
#to track the total number of latitudes available
nn=df['lat'].max()-df['lat'].min()#15
#Libraries to keep track of the patches and stages
D={}
D1={}
#extracting the data from the data frames  and storing according to the patch and stage. We start from the first to the last last patch (1:nn+1)  and when in each patch, we extract the number of species for  each life stage, the temperature for each patch (and compute the avaerage for the year)
for q in range(1,nn+2):
    #Juveniles for patch 33+q( since the min patch is 34, we will start with patch 34 --to the maximum
    D['J_patch'+ str(q)]=df.loc[(df['lat'] == 33+ q) & (df['lengthclass']=='smalljuv')]
    #total number of observations in each patch for each year
    n=len(D['J_patch'+ str(q)].year.values)
    # the total number of years of data available
    m=df['year'].max()-df['year'].min()
    #temperature readings when each species was caught in the patch
    Abun_TemJ=np.empty((m+1, 3))
    kJ=0
    kY=0
    kA=0
    for i in range (0, m+1):
        Abun_TemJ[i,0]=kJ
        DD=D['J_patch'+ str(q)]
        TT=DD.loc[(DD['year'] == 1980+i)]
        temp1=DD.loc[(DD['year'] == 1980+i)]
        print(temp1.temp_bottom.values)
        Abun_TemJ[i,1]=temp1.temp_bottom.values.mean()
        Abun_TemJ[i,2]=TT.NUMLEN.values.sum()
        
        kJ=kJ +1
    #After extracting teh temperature and calculating the mean value, we now save it
    D1['J_patch'+ str(q)]=Abun_TemJ
# now moving to Young Juveniles to perform teh same process as above
    D['Y_patch'+ str(q)]=df[(df['lat'] == 33+ q) & (df['lengthclass']=='largejuv')]
    n=len(D['Y_patch'+ str(q)].year.values)
    m=df['year'].max()-df['year'].min()
    Abun_TemY=np.empty((m+1, 3))
    for i in range (0, m+1):
        Abun_TemY[i,0]=kY
        DD=D['Y_patch'+ str(q)]
        TT=DD.loc[(DD['year'] == 1980+i)]
        temp1=DD.loc[(DD['year'] == 1980+i)]
        Abun_TemY[i,1]=temp1.temp_bottom.values.mean()
        Abun_TemY[i,2]=TT.NUMLEN.values.sum()
        kY=kY +1
    D1['Y_patch'+ str(q)]=Abun_TemY
#Next we move to Adult and perform teh same as above
    D['A_patch'+ str(q)]=df[(df['lat'] == 33+ q) & (df['lengthclass']=='adult')]
    Abun_TemA=np.empty((m+1, 3))
    for i in range (0, m+1):
        Abun_TemA[i,0]=kA
        DD=D['A_patch'+ str(q)]
        TT=DD.loc[(DD['year'] == 1980+i)]
        temp1=DD.loc[(DD['year'] == 1980+i)]
        Abun_TemA[i,1]=temp1.temp_bottom.values.mean()
        Abun_TemA[i,2]=TT.NUMLEN.values.sum()
        kA=kA +1
    D1['A_patch'+ str(q)]=Abun_TemA

 
