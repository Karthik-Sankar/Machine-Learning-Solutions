# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:20:04 2018

@author: Karthikeyan Sankar
"""

#Thompson Sampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
#implement Thompson Sampling
N = 10000
d = 10
total_reward = 0
ads_selected = []
number_of_rewards_1 = [0]*d
number_of_rewards_0 = [0]*d
for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(number_of_rewards_1[i]+1, number_of_rewards_0[i]+1)
        if(random_beta > max_random):
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if(reward == 1):
        number_of_rewards_1[ad] += 1
    elif(reward == 0):
        number_of_rewards_0[ad] += 1
    total_reward += reward
#visualising result
plt.hist(ads_selected)
plt.show()