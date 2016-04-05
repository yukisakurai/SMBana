import os
from pylab import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy import optimize
from scipy.interpolate import spline
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoLocator
from matplotlib.legend_handler import HandlerLine2D
import datetime
import argparse
import lib_smb as lib_smb
import lib_TS as lib_TS

T1 = np.load('20160330_YSTM.npz')['T']
B1 = np.load('20160330_YSTM.npz')['B']
HST1 = np.load('20160330_YSTM.npz')['HS_T']
T2 = np.load('20160331_YSTM.npz')['T']
B2 = np.load('20160331_YSTM.npz')['B']
HST2 = np.load('20160331_YSTM.npz')['HS_T']

# CANVAS
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.15,right=0.95)

# ================
# PLOTS
# ================
ax = fig.add_subplot(111)
xtitle = 'Temperature [K]'
ytitle = 'Magnetic Field [kG]'
plt.xlabel(xtitle, fontsize=24)
plt.ylabel(ytitle, fontsize=24)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.1, 0.5)

plt.plot(T1,
         HST1,
         'o',
         color='b',
         linewidth = 2.0,
         markersize = 2,
         label='cooling')

plt.plot(T2,
         HST2,
         'o',
         color='r',
         linewidth = 2.0,
         markersize = 2,
         label='heating')

# LEGEND
leg = plt.legend(loc='best',fontsize = 20)
leg.get_frame().set_linewidth(0.0)
for i in range(len(leg.legendHandles)):
    leg.legendHandles[i]._legmarker.set_markersize(10)

plt.grid()

plt.show()
