# compare optimal locations to places of high average concentration map
# compare optimal locations to places of high variance according to all possible states (32*144) when stacks emit average emissions
# my solution should probably be identical to the variance...
# However, if i add the cost with different sensor types, then it is different.

# generate all possible states (32*144) when stacks emit average emissions
# compare PEDs of optimal Vs. non-optimal solutions (randomly generated)

# generate weather states according to frequency and stacks emit all kinds of emissions (+-50% from average)
# compare PEDs of optimal Vs. non-optimal solutions (randomly generated)

import numpy as np
import csv
import timeit
# np.set_printoptions(precision=10)
import pandas as pd
import sys
sys.path.append('/Users/iditbela/Documents/Borg_python/Borg_downloaded_code/serial-borg-moea/')
from plugins.Python.borg import Borg
from matplotlib import pyplot as plt
from scipy.stats import ranksums

sys.path.append('/Users/iditbela/Documents/Borg_python/optimization_code/optimization_pycharm')
sys.path.append('/Users/iditbela/Documents/Borg_python/optimization_code/optimization_notebooks')
import conf_class_func
import data_preparation_functions
import objective_function

# (1) initialize the simulation
emissionRates = np.array([0.47,0.51,0.38,0.9,0.19])
sourceLoc = np.array([[200,300],[300,700],[650,400],[450,200],[200,500]])
num_S = np.shape(emissionRates)[0]
distanceBetweenSensors = 50
distanceFromSource = 50

Q_source, sensorArray, sensors_to_exclude = \
    data_preparation_functions.initializeSimulation(num_S, sourceLoc, emissionRates,
                                                    distanceBetweenSensors, distanceFromSource)


# (2) generate all possible states/maps (number of maps = 32*144) when stacks emit average emissions
# df = pd.read_pickle("/Users/iditbela/Documents/Borg_python/optimization_code/optimization_notebooks/WF_2004_2018_Hadera")
# # df is sorted
# # 144 states have certain probability different than zero
# numStates = df.loc[df.percent != 0,'s'].count()
# totalTotalField = np.zeros((441,32,numStates))
# for state in range(numStates):
# #     wf = df.iloc[state].percent
#     WD = df.iloc[state].WD_to_apply
#     WS = df.iloc[state].WS_to_apply
#     ASC = df.iloc[state].SC_to_apply
#     totalField, total_active = data_preparation_functions.calcSensorsReadings(Q_source, sensorArray, WD, WS, ASC)
#     totalTotalField[:,:,state] = totalField
#
# np.save('totalTotalField', totalTotalField)
# np.save('total_active', total_active)

# (2) compare PEDs of optimal Vs. non-optimal solutions (randomly generated) for all 144 weather states
totalTotalField = np.load('totalTotalField.npy')
total_active = np.load('total_active.npy')
nan_idx = np.load('nan_idx.npy')

# load optimal solutions
resultsPath = '/Users/iditbela/Documents/Borg_python/optimization_code/results_WF/'
objs = pd.read_csv(resultsPath+'objs.csv',header=None)
vars = pd.read_csv(resultsPath+'vars.csv',header=None)
objs.drop(0,axis=1,inplace=True)
vars.drop(0,axis=1,inplace=True)
objs.drop(0,axis=0,inplace=True)
vars.drop(0,axis=0,inplace=True)

vars = np.round(vars)
# fix values of PED
objs.iloc[:,1] = objs.iloc[:,1]*(-1)*1e9
# sort optimal solutions
sort_idx = np.argsort(objs.iloc[:,0])
sorted_vars = vars.iloc[sort_idx,:]
sorted_objs = objs.iloc[sort_idx,:]

sorted_objs.columns = ['numSens','mean_PED']
sorted_objs.reset_index(drop=True, inplace = True)
sorted_vars.reset_index(drop=True, inplace = True)

thr, dyR = 1,1
max_sensors = 80
# calc PEDs for optimal solutions
optimal_PEDs = np.zeros((np.size(range(2,max_sensors)),np.size(totalTotalField,2)))

for i, sol in enumerate(range(2,max_sensors)):
    ind_sol = np.argwhere(sorted_objs.numSens == sol)
    # ** where is the sensorIdx
    sensorIdx = np.argwhere(sorted_vars.iloc[ind_sol.ravel(),:].values.ravel()).ravel()

    for j, field in enumerate(range(np.size(totalTotalField,2))):
                totalField = np.squeeze(totalTotalField[:,:,field])
        # ** remove the nan rows **
        totalField = np.delete(totalField, nan_idx, axis=0)
        # ** calculate PED
        PEDs, scenario_pairs = data_preparation_functions.calcSensorsPED(totalField, total_active, sensorIdx, thr, dyR)
        [c, idx] = np.unique(np.sort(scenario_pairs[:, 2:4], axis=1), axis=0, return_inverse=True)
        min_PED = PEDs.groupby(idx).min()
        mean_PED = np.mean(min_PED)
        optimal_PEDs[i,j] = mean_PED
        print(i,j)

# np.save('optimal_PEDs', optimal_PEDs)

# calc PEDs for non-optimal solutions
n_iterations = 100
# P-VALUE IS NOT RELEVANT! IF ITS NOT THE SAME POPULATION IT DOESN'T MEAN THAT THE PEDS ARE STILL NOT HIGHER/LOWER!!!
# totalpVals = np.zeros((n_iterations,np.size(range(2,max_sensors))))
total_nonOptimal_PEDs = np.zeros((np.shape(optimal_PEDs)[0],np.shape(optimal_PEDs)[1],n_iterations))

for n in range(n_iterations):
    nonOptimal_PEDs = np.zeros((np.size(range(2,max_sensors)),np.size(totalTotalField,2)))

    for i, sol in enumerate(range(2,max_sensors)):
        # random sensorIdx
        sensorIdx = np.random.choice(np.arange(0, np.shape(totalTotalField)[0] - np.shape(nan_idx)[0]), sol)

        for j, field in enumerate(range(np.size(totalTotalField,2))):
            totalField = np.squeeze(totalTotalField[:,:,field])
            # ** remove the nan rows **
            totalField = np.delete(totalField, nan_idx, axis=0)
            # ** calculate PED
            PEDs, scenario_pairs = data_preparation_functions.calcSensorsPED(totalField, total_active, sensorIdx, thr, dyR)
            [c, idx] = np.unique(np.sort(scenario_pairs[:, 2:4], axis=1), axis=0, return_inverse=True)
            min_PED = PEDs.groupby(idx).min()
            mean_PED = np.mean(min_PED)
            nonOptimal_PEDs[i,j] = mean_PED
            # print(i,j)

    # pvals = []
    # for i, a in enumerate(range(2,max_sensors)):
    #     pvals.append(ranksums(optimal_PEDs[i,:],nonOptimal_PEDs[i,:]).pvalue)
    # pvals = np.array(pvals)
    # totalpVals[n,:] = pvals

    total_nonOptimal_PEDs[:,:,n] = nonOptimal_PEDs
    print(n)

# np.save('total_nonOptimal_PEDs', total_nonOptimal_PEDs)
# np.save('totalpVals', totalpVals)
# numSens = []
# for i, a in enumerate(range(2,max_sensors)):
#     numSens.append(a)
#     print(np.sum(totalpVals[:,i]>0.05)/100)
# numSens = np.array(numSens)
# numSens[np.argwhere(pvals>0.05).ravel()]

res = []
for i in range(10):
    temp = total_nonOptimal_PEDs[:,:,i]
    a = optimal_PEDs>temp
    plt.imshow(a)
    plt.show()
    # res.append(np.sum(a) / (np.size(a, 0) * np.size(a, 1)))
    # print(np.sum(a)/(np.size(a,0)*np.size(a,1)))

# # PED matrix (in notebook)
# totalField = np.load('weightedField.npy')
# sol = 9
# ind_sol = np.argwhere(sorted_objs.numSens == sol)
# # ** where is the sensorIdx
# sensorIdx = np.argwhere(sorted_vars.iloc[ind_sol.ravel(),:].values.ravel()).ravel()
#
# # ** remove the nan rows **
# totalField = np.delete(totalField, nan_idx, axis=0)
# # ** calculate PED
# PEDs, scenario_pairs = data_preparation_functions.calcSensorsPED(totalField, total_active, sensorIdx, thr, dyR)
# [c, idx] = np.unique(np.sort(scenario_pairs[:, 2:4], axis=1), axis=0, return_inverse=True)
# min_PED = PEDs.groupby(idx).min()*1e9
# temp = np.concatenate((c,min_PED.values),axis=1)
#
# z = np.zeros((6,6))
# x = np.arange(0,6)
# y = np.arange(0,6)
# for i in range(np.shape(temp)[0]):
#     z[int(temp[i,0]),int(temp[i,1])] = temp[i,2]
# w = np.transpose(z)
#
# fig, ax = plt.subplots()
# ax.pcolor(x,y,z+w)
# ax.set_title('thick edges')
# fig.tight_layout()
# plt.show()



# # boxplot compare optimal Vs. non-optimal
# fig = plt.figure()
# fig.suptitle('Solution comparison')
# ax = fig.add_subplot()
# # optimal
# bpOptimal = plt.boxplot(optimal_PEDs[0:40,:].transpose(), showfliers=False, positions = np.arange(1,41)-0.3, widths=0.25, patch_artist=True)
# plt.setp(bpOptimal['boxes'], color='blue')
# for patch in bpOptimal['boxes']:
#     patch.set_facecolor('blue')
#     patch.set_alpha(0.3)
# # non-optimal
# bpnonOptimal = plt.boxplot(total_nonOptimal_PEDs[0:40,:,0].transpose(), showfliers=False, positions = np.arange(1,41), widths=0.3, patch_artist=True)
# plt.setp(bpnonOptimal['boxes'], color='orange')
# for patch in bpnonOptimal['boxes']:
#     patch.set_facecolor('orange')
#     patch.set_alpha(0.3)
# ax.set_xticklabels(sorted_objs.iloc[0:40,0].values)
# # plt.grid()
# plt.show();

# Pareto front
fig = plt.figure()
plt.scatter(objs.iloc[:,0],objs.iloc[:,1])
plt.show();


df = pd.read_pickle("/Users/iditbela/Documents/Borg_python/optimization_code/optimization_notebooks/WF_2004_2018_Hadera")
# df is sorted
# 144 states have certain probability different than zero
numStates = df.loc[df.percent != 0,'s'].count()
WF = []
for state in range(numStates):
    WF.append(df.iloc[state].percent)
plt.scatter(np.arange(0,144),WF)
plt.show();

# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# look at the states and the performance of 1 random state
r = 50
i = 143
fig, ax = plt.subplots()
plt.scatter(np.arange(0,78), optimal_PEDs[:,i]*1e9)
plt.scatter(np.arange(0,78), total_nonOptimal_PEDs[:,i,r]*1e9)
plt.title('State: '+np.str(i)+' ,frequency: '+np.str(WF[i]))
# axins = inset_axes(ax, width=1.3, height=0.9, loc=2)
plt.show();

# is the frequency of the states related to the performance of the optimal solution?
numStates = df.loc[df.percent != 0,'s'].count()
numOfSens = 10
res = np.zeros(numStates)
for state in range(numStates):
    total_cnt = 0
    for r in range(100):
        temp = total_nonOptimal_PEDs[numOfSens-2,state,r]
        if not (optimal_PEDs[numOfSens-2,state]>temp):
            total_cnt += 1
    res[state]=total_cnt/100

plt.scatter(WF,res)
plt.show()


