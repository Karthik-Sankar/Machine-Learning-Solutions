# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:19:03 2018

@author: Karthikeyan Sankar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
#implement UCB
N = 10000
d = 10
total_reward = 0
ads_selected = []
num_selections = [0] * d
sums_rewards = [0] * d
for n in range(0,N):
    ad = 0
    max_upper = 0
    for i in range(0,d):
        if(num_selections[i] > 0):
            avg_reward = sums_rewards[i]/num_selections[i]
            delta_i = m.sqrt(3/2*m.log(n+1)/num_selections[i])
            upper_bound = avg_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_upper):
            max_upper = upper_bound
            ad = i
    ads_selected.append(ad)
    num_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_rewards[ad] += reward
    total_reward += reward
#visualising result
plt.hist(ads_selected)
plt.show()
