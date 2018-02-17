# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 18:40:11 2018
@author: Karthikeyan Sankar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')

X = dataset.iloc[:, [3, 5, 6, 7, 8, 9 ,10, 11, 12]]
y = dataset.iloc[:, -1]
 