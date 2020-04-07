######################################################################
#     This  script is the property of Pinsky's lab but you are free to  redistribute it and/or
#     modify it. It  uses  an ABC framework for estimating the parameters of a stage/size-structured process-based models of range dynamics using spatial abundance data. The procedure for ABC is well described in (Bertorelle et al. 2010).

######################################################################
from __future__ import print_function, division

import numpy as np
import math as math
import random as random
import sys
import copy as copy
from scipy.stats import norm
import statsmodels.api as sm
from scipy import stats
from numpy.linalg import inv
import pymc3
import scipy.stats as sst
from sklearn import preprocessing
#import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
############################################################################################
######################


def temp_dependence(temperature, Topt, width, kopt):
    """ compute growth rate as a function of temperature, were kopt is the optimal growth rate, Topt, optimal temperature, width, the standard deviation from the optimal temperature.
        """
    #theta = -width*(temperature-Topt)*(temperature-Topt) + kopt
    theta = kopt*np.exp(-0.5*np.square((temperature-Topt)/width))
    #theta = kopt*(1-(np.square((temperature-Topt)/width)))
    #theta=((temperature-Tmin)*(temperature-Tmax))/(((temperature-Tmin)*(temperature-Tmax))-(temperature-Topt))
    return theta
#############################################################################

def simulation_population(params):
    """ Takes in the initial population sizes and simulates the population size moving forward """
    # Set starting numbers for population and allocate space for population sizes
    N_JS = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    alpha=N_JS.copy()
    N_YS=N_JS.copy()
    N_AS=N_JS.copy()
    for x in range(0, cols):
        alpha[:,x]=temp_dependence(tempA[:,x],  params["Topt"], params["width"], params["kopt"])
    N_JS[0,:]=N_J[0,:]
    N_YS[0,:]=N_Y[0,:]
    N_AS[0,:]=N_A[0,:]
    L_bJ=params["L_inf"]-(params["L_inf"]-params["L_J"])*np.exp(alpha)
    L_bY=params["L_inf"]-(params["L_inf"]-params["L_Y"])*np.exp(alpha)
    g_J=(params["L_J"]-L_bJ)/(params["L_J"]-params["L_0"])
    g_Y=(params["L_Y"]-L_bY)/(params["L_Y"]-params["L_J"])
    for t in range(0,rows-1):
        #boundary values

        N_JS[t+1,0]=max(0,(1-params["m_J"])*N_JS[t,0]+(1-params["xi"])*N_AS[t,0]*np.exp(alpha[t,0]*(1-N_AS[t,0]/params["K"]))-g_J[t,0]*N_JS[t,0]+ params["xi"]*N_AS[t,1]*np.exp(alpha[t,1]*(1-N_AS[t,1]/params["K"])))
        N_JS[t+1,cols-1]=max(0,(1-params["m_J"])*N_JS[t,cols-1]+(1-params["xi"])*N_AS[t,cols-1]*np.exp(alpha[t,cols-1]*(1-N_AS[t,cols-1]/params["K"]))+ params["xi"]*N_AS[t,cols-2]*np.exp(alpha[t,cols-2]*(1-N_AS[t,cols-2]/params["K"]))-g_J[t,cols-1]*N_JS[t,cols-1])
        N_YS[t+1, 0] = max(0,(1-params["m_Y"])*N_YS[t,0]+g_J[t,0]*N_JS[t,0]-g_Y[t,0]*N_YS[t,0])
        N_YS[t+1, cols-1] = max(0,(1-params["m_Y"])*N_YS[t,cols-1]+g_J[t, cols-1]*N_JS[t,cols-1]-g_Y[t,cols-1]*N_YS[t,cols-1])
        
        N_AS[t+1, 0]=max(0,(1-params["m_A"])*N_AS[t,0]+g_Y[t,0]*N_YS[t,0]-params["xi"]*N_AS[t,0]+ params["xi"]*N_AS[t,1])
        N_AS[t+1, cols-1]=max(0,(1-params["m_A"])*N_AS[t,cols-1]+g_Y[t,cols-1]*N_YS[t,cols-1]-params["xi"]*N_AS[t,cols-1]+ params["xi"]*N_AS[t,cols-2])
        for x in range(1, cols-1):
            N_JS[t+1, x]=max(0,(1-params["m_J"])*N_JS[t,x]-g_J[t,x]*N_JS[t,x]+(1-2*params["xi"])*N_AS[t,x]*np.exp(alpha[t,x]*(1-N_AS[t,x]/params["K"]))+ params["xi"]*N_AS[t,x+1]*np.exp(alpha[t,x+1]*(1-N_AS[t,x+1]/params["K"]))+ params["xi"]*N_AS[t,x-1]*np.exp(alpha[t,x-1]*(1-N_AS[t,x-1]/params["K"])))
            N_YS[t+1, x] = max(0,(1-params["m_Y"])*N_YS[t,x]+g_J[t,x]*N_JS[t,x]-g_Y[t,x]*N_YS[t,x])
            N_AS[t+1, x]=max(0,(1-params["m_A"])*N_AS[t,x]+g_Y[t,x]*N_YS[t,x]-2*params["xi"]*N_AS[t,x]+ params["xi"]*N_AS[t,x+1]+params["xi"]*N_AS[t,x-1])
    return N_JS, N_YS, N_AS


##################################################################################################################################################################
#Caluclate summary statistics for the oberve data
def calculate_summary_stats(N_J, N_Y, N_A):
    """Takes in a matrix of time x place population sizes for each stage and calculates summary statistics"""
    time=range(T_FINAL)
    L_Q1=np.percentile(time, 5, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    L_Q=np.percentile(time, 25, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    M_Q=np.percentile(time, 50, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    M_Q1=np.percentile(time,60, axis=None, out=None, overwrite_input=False,interpolation='nearest')
    U_Q=np.percentile(time, 75, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q1=np.percentile(time, 80, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q2=np.percentile(time, 85, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q3=np.percentile(time, 90, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q4=np.percentile(time,95, axis=None, out=None, overwrite_input=False,interpolation='nearest')
    U_Q5=np.percentile(time, 97, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    U_Q6=np.percentile(time, 99, axis=None, out=None, overwrite_input=False, interpolation='nearest')
    SS_adult1=np.hstack((N_A[L_Q1], N_A[L_Q], N_A[M_Q],N_A[M_Q1], N_A[U_Q],N_A[U_Q1],N_A[U_Q2],N_A[U_Q3],N_A[U_Q4], N_A[U_Q5], N_A[U_Q6]))#np.std(N_A, axis=0), np.mean(N_A, axis=0), sst.skew(N_A, axis=0, bias=True),
    SS_young1=np.hstack((N_Y[L_Q1], N_Y[L_Q], N_Y[M_Q],N_Y[M_Q1], N_Y[U_Q],N_Y[U_Q1], N_Y[U_Q2], N_Y[U_Q3], N_Y[U_Q4], N_Y[U_Q5], N_Y[U_Q6]))#np.std(N_Y, axis=0), np.mean(N_Y, axis=0), bias=True),
    SS_juv1=np.hstack((N_J[L_Q1], N_J[L_Q], N_J[M_Q],N_J[M_Q1], N_J[U_Q],N_J[U_Q1], N_J[U_Q2], N_J[U_Q3], N_J[U_Q4], N_J[U_Q5], N_J[U_Q6]))#np.std(N_J, axis=0), np.mean(N_J, axis=0), axis=0, bias=True),
    #SS_adult1=np.hstack((N_A[L_Q1], N_A[L_Q], N_A[M_Q],N_A[M_Q1], N_A[U_Q]), N_A[U_Q4]))
    #SS_young1=np.hstack((N_Y[L_Q1], N_Y[L_Q], N_Y[M_Q],N_Y[M_Q1], N_Y[U_Q]), N_Y[U_Q4]))
    #SS_juv1=np.hstack(( N_Y[L_Q1], N_Y[L_Q], N_Y[M_Q],N_Y[M_Q1], N_Y[U_Q], N_Y[U_Q4]))
    #SS_adult1=np.hstack((N_A[L_Q], N_A[M_Q], N_A[U_Q]))
    #SS_young1=np.hstack((N_Y[L_Q], N_Y[M_Q], N_Y[U_Q]))
    #SS_juv1=np.hstack((N_Y[L_Q], N_Y[M_Q], N_Y[U_Q]))
    # SS_adult1=np.hstack((N_A))
    #SS_young1=np.hstack((N_Y))
    #SS_juv1=np.hstack(( N_J))
    return SS_adult1, SS_young1, SS_juv1
##############################################################################################################################

def small_percent(vector, percent):
    """ Takes a vector and returns the indexes of the elements within the smallest (percent) percent of the vector"""
    sorted_vector = sorted(vector)
    cutoff = math.floor(len(vector)*percent/100) # finds the value which (percent) percent are below
    indexes = []
    
    cutoff = int(cutoff)

    for i in range(0,len(vector)):
        if vector[i] <= sorted_vector[cutoff]: # looks for values below the cutoff
            indexes.append(i)
    return indexes, sorted_vector[cutoff]


def z_score(x):
    """Takes a list and returns a 0 centered, std = 1 scaled version of the list"""
    st_dev = np.std(x,axis=0)
    mu = np.mean(x,axis=0)
    rescaled_values = []
    for element in range(0,len(x)):
        rescaled_values[element] = (x[element] - mu) / st_dev

    return rescaled_values

############################
# this function transform the paramters using logit function. the aim is to ensure that we do not end up with a parameter out of the prior
def do_logit_transformation(library, param_bound):
    for i in range(len(library[0,:])):
        library[:,i]=(library[:,i]-param_bound[i,0])/(param_bound[i,1]-param_bound[i,0])
        library[:,i]=np.log(library[:,i]/(1-library[:,i]))
    return library
###########################
#this function back transform parameter values
def do_ivlogit_transformation(para_reg, param_bound):
    for i in range(len(library[0,:])):
        para_reg[:,i]=np.exp(para_reg[:,i])/(1+np.exp(para_reg[:,i]))
        para_reg[:,i]=para_reg[:,i]*(param_bound[i,1]-param_bound[i,0])+param_bound[i,0]
    return para_reg
############################
#
def do_kernel_ridge(stats, library, param_bound):
    "This function transforms the accepted parameter values using local linear regression"
    X = sm.add_constant(stats)
    Y=library
    clf     = KernelRidge(alpha=1.0, kernel='rbf', coef0=1)
    resul   = clf.fit(X, Y)
    resul_coef=np.dot(X.transpose(), resul.dual_coef_)
    coefficients =resul_coef[1:]
    para_reg   =Y- stats.dot(coefficients)
    para_reg=do_ivlogit_transformation(para_reg, param_bound)
    parameter_estimate = np.average(para_reg, axis=0)
    HPDR=pymc3.stats.hpd(para_reg)
    return parameter_estimate, HPDR
##############################################################################################
def do_rejection(library):
    "This function performs rejection ABC"
    parameter_estimate = np.average(library, axis=0)
    HPDR=pymc3.stats.hpd(library)
    return parameter_estimate, HPDR

    #####################################################################################
#return all the observe summaery statitics (OS)and simulated summary statistics (SS) in a matrix with first row corresponding to OS and the rest of the rows to SS
def run_sim():
    PARAMS_ABC = {}# copies parameters so new values can be generated;
    #an array for the parameters that wil be generated using ABC
    param_save = []
    #Calculate summary stastistics for the observed data
    SS_adult, SS_young, SS_juv= calculate_summary_stats(N_J, N_Y, N_A)
    
    SO=np.hstack((SS_adult, SS_young, SS_juv))
    Obs_Sim=np.zeros((NUMBER_SIMS+1,len(SO)))
    Obs_Sim[0,:]=SO
    #simulaate candidate parameters
    for i in range(0,NUMBER_SIMS):
        L_0_theta    = np.random.uniform(0,4)#np.random.normal(0.4,0.3) #np.random.beta(2,2)
        L_inf_theta    =np.random.uniform(100, 180)
        L_J_theta=34
        L_Y_theta=68
        #np.random.uniform(0,1)#np.random.beta(2,2)
        Topt_theta =np.random.uniform(6,15)#np.random.normal(6.5,2) #np.random.uniform(1,12) #np.random.lognormal(1,1)
        width_theta  =np.random.uniform(1,2)#np.random.normal(2,1)
        ##np.random.lognormal(1,1)
        kopt_theta    =np.random.uniform(0.1,1)#np.random.normal(0.5,0.4)# np.random.u(0,1)
        xi_theta     =np.random.uniform(0,0.5/2)#np.random.normal(0.1,0.09) #np.random.normal(0,1)#np.random.normal(0,0.5)
        #r_theta     =np.random.uniform(0,1)
        m_J_theta    =np.random.uniform(0,0.1)#np.random.normal(0.04,0.04) # #np.random.beta(2,2)
        m_Y_theta    =np.random.uniform(0,0.1)#np.random.normal(0.05,0.04) #np.random.uniform(0,1) #np.random.beta(2,2)
        m_A_theta    =np.random.uniform(0,0.1)#np.random.normal(0.05,0.05)# np.random.uniform(0,1)#np.random.beta(2,2)
        K_theta= np.random.uniform(1,1000)
        PARAMS_ABC["L_0"]    = L_0_theta # s
        PARAMS_ABC["L_inf"]    =L_inf_theta
        PARAMS_ABC["L_J"]    =  L_J_theta
        PARAMS_ABC["L_Y"]    = L_Y_theta
        PARAMS_ABC["Topt"] = Topt_theta
        PARAMS_ABC["width"]  = width_theta
        PARAMS_ABC["kopt"]    = kopt_theta
        PARAMS_ABC["xi"]     = xi_theta
        #PARAMS_ABC["r"]     = r_theta
        PARAMS_ABC["m_J"]    = m_J_theta
        PARAMS_ABC["m_Y"]    = m_Y_theta
        PARAMS_ABC["m_A"]    = m_A_theta
        PARAMS_ABC["K"]    = K_theta
        # Simulate population for new parameters
        N_J_sim, N_Y_sim, N_A_sim = simulation_population(PARAMS_ABC)
        
        # Calculate the summary statistics for the simulation
        Sim_SS_adult, Sim_SS_young, Sim_SS_juv= calculate_summary_stats(N_J_sim, N_Y_sim, N_A_sim)
        SS=np.hstack((Sim_SS_adult, Sim_SS_young, Sim_SS_juv))
        Obs_Sim[i+1,:]=SS
        saving candidate parameter values
        param_save.append([L_0_theta,L_inf_theta, Topt_theta, width_theta, kopt_theta, xi_theta, m_J_theta,m_Y_theta, m_A_theta, K_theta])
    
    return np.asarray(param_save), Obs_Sim
#########################################################################################
#return all  all parameters (library) and simulated  NSS (stats) corresponding to d ≤ δ(eps).
def compute_scores(dists, param_save, difference,Sim_SS):
    eps=0.1
    library_index, NSS_cutoff = small_percent(dists, eps)
    n                = len(library_index)
    library = np.empty((n, param_save.shape[1]))
    stats            = np.empty((n, difference.shape[1]))
    stats_SS            = np.empty((n, difference.shape[1]))
    for i in range(0,len(library_index)):
        j = library_index[i]
        library[i] = param_save[j]
        stats[i]   = difference[j]
        stats_SS[i]   = Sim_SS[j]
    return library, stats, NSS_cutoff, library_index, stats_SS
##########################################################################################
#computes weights for local regression

def compute_weight(kernel,t, eps, index):
     weights=np.empty(len(index))
     if (kernel == "epanechnikov"):
         for i in range(0,len(library_index)):
             j = library_index[i]
     #weights[i]= (1. - (t[j] / eps)**2)
             weights[i]=(1. - (t[j] / eps)**2)
     elif(kernel == "rectangular"):
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]=t[j] / eps
     elif (kernel == "gaussian"):
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]= 1/np.sqrt(2*np.pi)*np.exp(-0.5*(t[j]/(eps/2))**2)
            
     elif (kernel == "triangular"):
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]= 1 - np.abs(t[j]/eps)
     elif (kernel == "biweight"):
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]=(1 - (t[j]/eps)**2)**2
     else:
          for i in range(0,len(library_index)):
              j = library_index[i]
              weights[i]= np.cos(np.pi/2*t[j]/eps)
     return weights
###########################################################################################

def sum_stats(Obs_Sim, param_save):
    dists = np.zeros((NUMBER_SIMS,1))
    #Obs_Sim_scale=np.nan_to_num(sst.zscore(Obs_Sim, axis=0,ddof=1),copy=True)
    Obs_Sim_scale=np.nan_to_num(preprocessing.normalize(Obs_Sim, axis=0),copy=True)
    #Substract each row of teh array from row 1
    Sim_SS=Obs_Sim_scale[1:NUMBER_SIMS+1,: ]
    Obs_SS=Obs_Sim_scale[0,:]
    difference=Obs_Sim_scale[1:NUMBER_SIMS+1,: ]-Obs_Sim_scale[0,:]
    #c=np.std(Obs_Sim_scale[1:NUMBER_SIMS+1,: ], axis=1)
    # compute the norm 2 of each row
    dists = np.linalg.norm(difference, axis=1)
    library, stats, NSS_cutoff, library_index, stats_SS = compute_scores(dists, param_save, difference,Sim_SS)
    # print(library)
    return library, dists, stats,stats_SS,   NSS_cutoff, library_index
###################################################################################################

def do_regression(library, stats, PARAMS):

    # REJECTION
    print('\nDo a rejection ABC:')
    do_rejection(library, PARAMS)
    do_local_linear(stats, library, weights,KK)
    #print('\nStats:', stats.shape)
    #print('\nStats:', stats)
    #print('\nLibar:', library.shape)
    #print('\nLibar:', library)

    do_kernel_ridge(stats, library)
    do_ridge(stats, library)
##################################################################################################
def do_goodness_fit(result,HPDR, actual, n, i):
    for j in range(0,n):
        if HPDR[j][0]<=actual[j]<=HPDR[j][1]:
           coverage[i,j]=1
        else:
           coverage[i,j]=0
    resultsbias[i,:] = (result - actual)/actual
    return coverage,resultsbias

#############################################################################################
#main script starts from here
if __name__ == '__main__':
    # Import the abundance data and data for the other variables e.g temperature
    import groundfish_training
    # the total number of generations
    T_FINAL = len(groundfish_training.D1['J_patch1'][:,0])
    #We simulate 20000 sets of parameters for for ABC, using non informatives priors (uniform priors
    NUMBER_SIMS = 20000
    #no of patches
    no_patches=12
    rows=T_FINAL
    cols=no_patches
    # creating an array to store the number of juveniles, young juvenils and adults in each patch
    N_J=np.ndarray(shape=(rows, cols), dtype=float, order='F')
    N_Y=np.ndarray(shape=(rows, cols), dtype=float, order='F')
    N_A=np.ndarray(shape=(rows, cols), dtype=float, order='F')
    tempA = np.ndarray(shape=(rows, cols), dtype=float, order='F')
    #storing data (secies abundance and temeprature time series data ) in the created arrays
    for q in range(1,no_patches+1):
        i=q-1
        p=q
        N_J[:,i]=groundfish_training.D1['J_patch'+ str(p)][:,2]
        N_Y[:,i]=groundfish_training.D1['Y_patch'+ str(p)][:,2]
        N_A[:,i]=groundfish_training.D1['A_patch'+ str(p)][:,2]
        tempA[:,i]=groundfish_training.D1['A_patch'+ str(p)][:,1]
    #running ABC. See the function for details. returns all the observe summary statitics (OS)and simulated summary statistics (SS) in a matrix with first row corresponding to OS and the rest of the rows to SS as well as the parameter values that led to the simulated summary statistics.
    param_save, Obs_Sim         = run_sim()
    ######################################################################################################
#normalize the rows of Obs_sim to have NOS in row 1 and NSS in the remaining rows. Substract rows i=2:NUMBER_SIMS from row 1 of Obs_sim (whic contain OS).Compute the eucleadean distance (d) between NSS and NOS then use it along side tolerance (δ), to determine all parameters and NSS corresponding to d ≤ δ.Choose δ such that δ × 100% of the NUMBER_SIMS simulated parameters and NSS are selected. retain the parameters that made this threshold (library), the weights ot be used in local linear regression and the NSS that meets the threshold (stats)
    library, dists, stats,stats_SS,  NSS_cutoff, library_index   = sum_stats(Obs_Sim, param_save)
# performing rejectio ABC. Note that if UMBER_SIMS is big enough, but rejection and regression ABC leads to teh same results.
    result, HPDR=do_rejection(library)
    print('see the results below')
    print('Estimates from rejection is:', result)
    print('Estimated HPDR from rejection is :', HPDR)
# Next we have regression ABC, perform it if only you are not performing rejection ABC above. Gives better results for NUMBER_SIMS small. I have commented it.
#library_reg=do_logit_transformation(library, param_bound)LJ=34, Ly=68, Linf=200
        #result_reg, HPDR_reg=do_kernel_ridge(stats, library_reg, param_bound)
    PARAMS1={}
    print(result[2])
    PARAMS1 = {"L_0":result[0] , "L_inf": result[1],"L_J": 68,"L_Y": 34, "Topt": result[2], "width": result[3], "kopt": result[4],"xi":result[5], "m_J": result[6], "m_Y":result[7] , "m_A": result[8], "K": result[9]}

    N_J1, N_Y1, N_A1 = simulation_population(PARAMS1)

#Importing a file call plot to plot the results.
    import plot
    print('i just imported a plot')
    for q in range(1,no_patches+1):
        i=q-1
        p=q
        plot.do_realdata(N_J1[:,i], N_J[:,i],  'J_abun_rej'+ str(p))
        #plot.do_scatter(N_J1[:,i], N_J[:,i],  'J_abun_scatter'+ str(p))
        plot.do_realdata(N_Y1[:,i], N_Y[:,i],  'Y_abun_rej'+ str(p))
        #plot.do_scatter(N_Y1[:,i], N_Y[:,i],  'Y_abun_scatter'+ str(p))
        plot.do_realdata(N_A1[:,i], N_A[:,i],  'A_abun_rej'+ str(p))
#plot.do_scatter(N_A1[:,i], N_A[:,i],  'A_abun_scatter'+ str(p))
################################################################
# plot the figures below if you willl like to plot the heatmap
    NJ1=N_J1.transpose()
    NJ=N_J.transpose()
    NY1=N_Y1.transpose()
    NY=N_Y.transpose()
    NA1=N_A1.transpose()
    NA=N_A.transpose()
    print(NJ1.shape)
    ax=sns.heatmap(NJ1, cmap="Greys", xticklabels=True, yticklabels=True,  cbar_kws={'label': 'Abundance'})
    plt.xlabel("Year")
    plt.ylabel("Latitude")
    ax.set_xticklabels(pd.Series(range(1980, 2012)))
    ax.set_yticklabels(pd.Series(range(34, 46)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.savefig("sim_J.png", bbox_inches='tight')
    plt.close()
#############################################################
    ax = sns.heatmap(NJ, cmap="Greys",  xticklabels=True, yticklabels=True, cbar_kws={'label': 'Abundance'})
    plt.xlabel("Year")
    plt.ylabel("Latitude")
    ax.set_xticklabels(pd.Series(range(1980, 2012)))
    ax.set_yticklabels(pd.Series(range(34, 46)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.savefig("Obs_J.png", bbox_inches='tight')
    plt.close()
##########################################################
    ax = sns.heatmap(NY1, cmap="Greys", xticklabels=True, yticklabels=True,  cbar_kws={'label': 'Abundance'})
    plt.xlabel("Year")
    plt.ylabel("Latitude")
    ax.set_xticklabels(pd.Series(range(1980, 2013)))
    ax.set_yticklabels(pd.Series(range(34, 46)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.savefig("Sim_Y.png", bbox_inches='tight')
    plt.close()
##########################################################
    ax = sns.heatmap(NY, cmap="Greys", xticklabels=True, yticklabels=True,  cbar_kws={'label': 'Abundance'})
    plt.xlabel("Year")
    plt.ylabel("Latitude")
    ax.set_xticklabels(pd.Series(range(1980, 2013)))
    ax.set_yticklabels(pd.Series(range(34, 46)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.savefig("Obs_Y.png", bbox_inches='tight')
    plt.close()
############################################################
    ax = sns.heatmap(NA1, cmap="Greys", xticklabels=True, yticklabels=True, cbar_kws={'label': 'Abundance'})
    plt.xlabel("Year")
    plt.ylabel("Latitude")
    ax.set_xticklabels(pd.Series(range(1980, 2013)))
    ax.set_yticklabels(pd.Series(range(34, 46)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.savefig("Sim_A.png", bbox_inches='tight')
    plt.close()
############################################################
    ax = sns.heatmap(NA, cmap="Greys",  xticklabels=True, yticklabels=True, cbar_kws={'label': 'Abundance'})
    plt.xlabel("Year")
    plt.ylabel("Latitude")
    ax.set_xticklabels(pd.Series(range(1980, 2013)))
    ax.set_yticklabels(pd.Series(range(34, 46)))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.figure.savefig("Obs_A.png", bbox_inches='tight')
    plt.close()

