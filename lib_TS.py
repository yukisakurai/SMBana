import os
import math
import datetime
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy import optimize
from scipy.interpolate import spline
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoLocator
import argparse
import inspect

def CalibSiPD(volt,serial):

    Z = volt # array of voltages
    T = np.array([]) # array of temprature

    # Read Calibration Data
    filename = os.getcwd() + '/CalibData/' + str(serial) + '/' + serial + '.cof.txt'
    if not os.path.exists(filename):
        print 'error : %s does not exist' % filename
        exit()
    f = open(filename,'r')
    data = f.read()
    data = data.split('FIT RANGE:')
    num_range = int(data[0].split(':')[1].strip())
    data = data[1:] # remove string of 'number of fit range'

    for i in range(len(Z)):
        temp = 0
        Vu = 0
        Vl = 0
        ZL = 0
        ZU = 0
        A = np.zeros(10)
        range_check = 0
        for ran in range(num_range):
            lines = data[ran].split('\n')
            lines = lines[1:] # remove range number

            # Fit range select
            for line in lines:
                if not (line.find('Lower Voltage limit') == -1): Vl = float(line.split(':')[1].strip())
                if not (line.find('Upper Voltage limit') == -1): Vu = float(line.split(':')[1].strip())
            if not ((Z[i] < Vu) and (Z[i] >= Vl)): continue
            range_check += 1
            for line in lines:
                if not (line.find('Zlower') == -1): ZL = float(line.split(':')[1].strip())
                if not (line.find('Zupper') == -1): ZU = float(line.split(':')[1].strip())
                for j in range(10):
                    if not (line.find('C(' + str(j) + ')')):  A[j] = float(line.split(':')[1].strip())

        if range_check != 1:
            print 'Invalid input voltage : V = %f' % Z[i]
            exit()

        k = ((Z[i] - ZL) - (ZU - Z[i])) / (ZU - ZL)

        for j in range(10): temp += A[j] * math.cos(j*math.acos(k))
        T = np.append(T,temp)

    return T

def TempCernox(SerialNumber,ResistArr):

    Thr = np.load('./CalibData/' + SerialNumber + '.npz')['THR']
    Param = np.load('./CalibData/' + SerialNumber + '.npz')['PARAM']
    T = np.array([])
    for i in range(len(ResistArr)):
        idx = -1
        if ResistArr[i] <= Thr[0] : idx = 0
        if ResistArr[i] > Thr[1] and ResistArr[i] <= Thr[2] : idx = 1
        if ResistArr[i] > Thr[2] and ResistArr[i] <= Thr[3] : idx = 2
        if ResistArr[i] > Thr[3] : idx = 3
        T = np.append(T,np.polyval(Param[idx],ResistArr[i]))

    return T

####

# # CANVAS
# fig = plt.figure(figsize=(10, 8))
# fig.subplots_adjust(left=0.15,right=0.95)

# # ================
# # PLOTS
# # ================
# ax = fig.add_subplot(111)
# ytitle = 'Temperature [K]'
# xtitle = r'Resistance [$\Omega$]'

# plt.xlabel(xtitle, fontsize=24)
# plt.ylabel(ytitle, fontsize=24)
# ax.xaxis.set_label_coords(0.5, -0.1)
# ax.yaxis.set_label_coords(-0.1, 0.5)

# x = np.linspace(100,10000,1000)
# plt.plot(x,
#          TempCernox('x74355',x),
#          '-',
#          color='r',
#          linewidth = 2.0)
# plt.show()
