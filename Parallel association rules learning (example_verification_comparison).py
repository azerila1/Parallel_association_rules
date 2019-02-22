#!/usr/bin/env python
# coding: utf-8

# Editor | Date | Comment
# --- | --- | ---
# Alireza Ranjbar | 20.10.2018 | Initial version

# In[1]:


import sys
import numpy as np
import pandas as pd
import os
import ast
import time
import matplotlib.pyplot as plt
from importlib import reload
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

# For calculating frequent patterns:
from mlxtend.frequent_patterns import apriori
# The original(serial computing) version of the association_rules learning:
from mlxtend.frequent_patterns import association_rules
# the modified version of the association_rules module source code from mlxtend for compatibility to be 
# used with multiprocessing.pool:
import MP_association_rules


# [A brief explanation about multiprocessing.pool](https://www.ellicium.com/python-multiprocessing-pool-process/)

# In[3]:


location=r'\\xxxxx'
os.chdir(location)

# for apriori function (run_assoc_rules_learning) :
train_data = pd.read_csv("xxx.csv", delimiter=';')
report_filter='xxxx'
filter_value='xxxx'
sortby='support'
Filtered_Data = train_data[train_data[report_filter]==filter_value] 

freqent_patterns = apriori(Filtered_Data.iloc[:,5:],
                           min_support=0.5,
                           use_colnames=True,
                           max_len=4)


# In[24]:


def make_subsets(main_set_df, n_subset):
    """
    splits the input DataFrame into a list of n_subset DataFrame
    """
    subset_size = int(len(main_set_df)/n_subset)
    list_of_subsets = []
    for i in range(n_subset-1):   
        list_of_subsets.append(main_set_df[(i)*subset_size:(i+1)*subset_size])
    list_of_subsets.append(main_set_df[(i+1)*subset_size:])
    return list_of_subsets


# In[38]:


def parallel_association_rules(freqent_patterns,
                               n_parallel_branch=mp.cpu_count(),
                               metric="confidence",
                               min_threshold=0.7):
    
    list_of_subsets = make_subsets(main_set_df=freqent_patterns,
                                   n_subset=n_parallel_branch)
    with mp.Pool(processes = n_parallel_branch) as pool:
        results = pool.map(partial
                             (
                                 MP_association_rules.association_rules_MP,
                                 main_set=freqent_patterns,
                                 metric=metric,
                                 min_threshold=min_threshold,
                                 support_only=False),
                                 list_of_subsets
                             ) 
        pool.terminate
    pool.close(); pool.join()
    Final_MP_results = results[0]
    for i in np.arange(1,len(results)):
        Final_MP_results = pd.concat([Final_MP_results,results[i]],ignore_index=True)   
    # returns the equivalent result of the mlxtend association_rules module but computed in parallel
    return Final_MP_results


# ### Test

# In[40]:


parallel_results = parallel_association_rules(freqent_patterns, n_parallel_branch=2, metric="confidence",
                                             min_threshold=0.7)

serrial_results = association_rules(freqent_patterns, metric="confidence", min_threshold=0.7)
serrial_results['antecedents'] = serrial_results['antecedents'].apply(lambda x: list(x))
serrial_results['consequents'] = serrial_results['consequents'].apply(lambda x: list(x))


# ### Verifying equality of the parallel version out put and that of serial version

# In[80]:


# conversion of lists to set, since their order does not matter
serrial_results['antecedents'] = serrial_results['antecedents'].apply(lambda x: set(x))
serrial_results['consequents'] = serrial_results['consequents'].apply(lambda x: set(x))
parallel_results['antecedents'] = parallel_results['antecedents'].apply(lambda x: set(x))
parallel_results['consequents'] = parallel_results['consequents'].apply(lambda x: set(x))


# In[79]:


# making sure the lenght of both dataframes are the same
# and all the row in one is found and unique in the other

assert len(parallel_results)==len(serrial_results)
for index, row in parallel_results.iterrows(): 
    matched_rule = serrial_results[serrial_results==row].dropna()
    if len(matched_rule) != 1:
        print('row with index '+ str(index)+ ' in parallel_results does match with any row in serrial results')
        break


# # Comparison of the speeds

# In[47]:


report_filter='xxxx'
filter_value='xxxx'
sortby='support'


Filtered_Data=train_data[train_data[report_filter]==filter_value]   
MP_times=[]
mlx_times=[]
apriori_length=[]
output_length=[]
for minsup in np.flip(np.arange(0.08,0.5,0.01)):
    
    freqent_patterns = apriori(Filtered_Data.iloc[:,5:],
                                                       min_support=minsup,
                                                       use_colnames=True,
                                                       max_len=None)



    t=time.time() 
    Final_MP_results = Parallel_association_rules(
                                                   freqent_patterns,
                                                   n_parallel_branch=mp.cpu_count(),
                                                   metric="confidence",
                                                   min_threshold=0.7,
                                                   sortby='conviction'
                                                  )

    MP_times.append(time.time()-t)
    #print('Parallel computation took: '+str(time.time()-t)+ ' seconds')


    t=time.time()
    assoc_rules = association_rules(freqent_patterns,
                                   metric="confidence",
                                   min_threshold=0.7)

    assoc_rules['antecedents'] = assoc_rules['antecedents'].apply(lambda x: list(x))
    assoc_rules['consequents'] = assoc_rules['consequents'].apply(lambda x: list(x))
    mlxtend_result=assoc_rules.sort_values(by=['conviction'] ,ascending=False)
    mlxtend_result.index=range(len(mlxtend_result))
    mlx_times.append(time.time()-t)


    
    apriori_length.append(len(freqent_patterns))
    output_length.append(len(assoc_rules))
    #print('Serrial computation took: '+str(time.time()-t)+ ' seconds')
    print('%.2f' %minsup+':'+str(len(freqent_patterns))+' ____ '+str(len(assoc_rules))+':'+str(len(Final_MP_results)))


# The below plots show the computation time of the parallel and original serrial implementation of the mlxtend association rule function. As shown, until ~3500 input number of frequent patterns(equvalent to ~170,000 generated output rules), the parallel implementation takes longer time to finish. Yet, for larger number of frequent patterns as input and more number of rules to be generated, the parallel implementation takes over and finish computation faster.

# In[54]:


plt.figure(figsize=[15,8])
plt.plot(apriori_length,mlx_times,label='mlxtends original serrial implementation',color='blue')
plt.plot(apriori_length,MP_times[:41],label='Parallel implementation of mlxtend',color='red')
plt.legend()
plt.xlabel('number of frequent patterns as input')
plt.ylabel('Computation time')


# In[49]:


plt.figure(figsize=[15,8])
plt.plot(apriori_length[:35],mlx_times[:35],label='mlxtends original serrial implementation',color='blue')
plt.plot(apriori_length[:35],MP_times[:35],label='Parallel implementation of mlxtend',color='red')
plt.legend()
plt.xlabel('number of frequent patterns as input')
plt.ylabel('Computation time')


# ##### Checking if the speed and the number of association rules to be made corrolate with each other

# In[21]:


report_filter = 'xxx'
filter_value ='xx'
sortby = 'xx'


Filtered_Data = train_data[train_data[report_filter]==filter_value]   
freqent_patterns = apriori(Filtered_Data.iloc[:,5:],
                           min_support=0.15,
                           use_colnames=True,
                           max_len=None)
MP1_times=[]
Final_MP_results1_length = []
for conf in np.flip(np.arange(0,1,0.005)):
    
    t=time.time() 
    Final_MP_results1 = Parallel_association_rules(
                                                   freqent_patterns,
                                                   n_parallel_branch=mp.cpu_count(),
                                                   metric="confidence",
                                                   min_threshold=conf,
                                                   sortby='conviction'
                                                  )
    MP1_times.append(time.time()-t)

    
    Final_MP_results1_length.append(len(Final_MP_results1))
    
    #print('Serrial computation took: '+str(time.time()-t)+ ' seconds')
    print('%.2f' %conf+':'+str(len(Final_MP_results1))+':'+str(time.time()-t))


# In[43]:


plt.figure(figsize=[15,8])

plt.plot(np.flip(np.arange(0,1,0.005)),[x/60 for x in MP1_times],label='Computational time',color='blue')
plt.plot(np.flip(np.arange(0,1,0.005)),[x/10000000 for x in Final_MP_results1_length],label='Number of rules / 10^7',color='red')
plt.legend()
plt.xlabel('Confidence')
plt.ylabel('Computation time (minutes) \n Number of rules/10^7')

