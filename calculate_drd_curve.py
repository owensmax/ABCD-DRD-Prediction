#set working directory
import os
os.chdir('/home/max/Documents/DRD/final_analyses_for_github')

from loading_final import get_setup_ML, get_cat_vars, load_data
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from BPt import *
import numpy as np

#load in data
data_all=load_data(get='all_noqc',val_check=3)
dat=data_all[0]

##############claculate DRD curve##########
mean1 = dat['ddis_scr_val_indif_point_6h'].mean()
mean2 = dat['ddis_scr_val_indif_pnt_1da'].mean()
mean3 = dat['ddis_scr_val_indif_pnt_1week'].mean()
mean4 = dat['ddis_scr_val_indif_pnt_1mth'].mean()
mean5 = dat['ddis_scr_val_indif_pnt_3mth'].mean()
mean6 = dat['ddis_scr_val_indif_pnt_1yr'].mean()
mean7 = dat['ddis_scr_val_indif_pnt_5yr'].mean()

means = [mean1, mean2, mean3, mean4, mean5, mean6, mean7]
names = ['6 hours', '1 day', '1 week', '1 month', '3 months', '1 year', '5 years']
delays = [.25, 1, 7, 30, 90, 365, 1825]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(delays, means, color = 'red' , marker='o', linewidth=3, markeredgewidth=3)
plt.title('Discounting Curve', fontsize = 24)
plt.xlabel('Delay in Days', fontsize = 22)
plt.ylabel('Indifference Point', fontsize = 22)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

for x,y,z in zip(delays, means, names):
    if x == 1:
        n = -7
        m = -7
    else:
        n = 0
        m = 0
    if x == 1825:
        n = -50
        m = 20
    plt.annotate(z, # this is the text
                 (x, y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(45+n,m),
                 fontsize=15, # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.savefig('Discounting_curve.tiff', dpi = 500, bbox_inches='tight')
plt.show()

