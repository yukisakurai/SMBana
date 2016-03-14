import os
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
import lib_smb as lib_smb

parser = argparse.ArgumentParser(description='description : This is plotting macro for driver RPM')
parser.add_argument('-f', '--file', nargs=1, type=str, help=' : input file name')
args = parser.parse_args()

dir = '/Users/' + getpass.getuser() + '/cernbox/LiteBIRD/analysis/data/TM/'
name = '20160121_TM'
if args.file:
    path = dir + name +'.txt'
    if os.path.exists(path):
        name = args.file[0]
    else:
        print 'error: input file does not exist'
        sys.exit()

fname = dir + name +'.txt'
constant_file = dir + 'temprature_constant.txt'
savename = 'plot/' + name + '.pdf'

Time = []
ch1 = []
ch2 = []
ch3 = []
ch4 = []
ch5 = []
ch6 = []
ch7 = []
ch8 = []

param = []

f = open(constant_file,'r')
i=0
for line in f.readlines():
    itemList = line[:-1].split(' ')
    temp = [itemList[1],itemList[2]]
    param.append(temp)
    i+=1

for line in open(fname, 'r'):
    itemList = line[:-1].split('\t')
    date = itemList[0].split('/')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    time = itemList[1].split(':')
    hour = int(time[0])
    min = int(time[1])
    sec = int(time[2])

    ch1.append(float(itemList[2]))
    ch2.append(float(itemList[3]))
    ch3.append(float(itemList[4]))
    ch4.append(float(itemList[5]))
    ch5.append(float(itemList[6]))
    ch6.append(float(itemList[7]))
    ch7.append(float(itemList[8]))
    ch8.append(float(itemList[9]))

    Datetime = datetime.datetime(year, month, day, hour=hour, minute=min)
    Time.append(Datetime)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
xlabel("Time", fontsize=20)
ylabel("Temperature [K]", fontsize=20)
ax.xaxis.set_label_coords(0.5, -0.15)
ax.yaxis.set_label_coords(-0.1, 0.5)

plt.plot(Time,ch1,'-o',markersize=1,markerfacecolor='b',markeredgecolor='b',label='ch1')
plt.plot(Time,ch2,'-o',markersize=1,markerfacecolor='g',markeredgecolor='g',label='ch2')
plt.plot(Time,ch3,'-o',markersize=1,markerfacecolor='r',markeredgecolor='r',label='ch3')
plt.plot(Time,ch4,'-o',markersize=1,markerfacecolor='c',markeredgecolor='c',label='ch4')
plt.plot(Time,ch5,'-o',markersize=1,markerfacecolor='m',markeredgecolor='m',label='ch5')
plt.plot(Time,ch6,'-o',markersize=1,markerfacecolor='y',markeredgecolor='y',label='ch6')
plt.plot(Time,ch7,'-o',markersize=1,markerfacecolor='y',markeredgecolor='y',label='ch6')
plt.plot(Time,ch8,'-o',markersize=1,markerfacecolor='y',markeredgecolor='y',label='ch6')

days = mdates.AutoDateLocator()
daysFmt = mdates.DateFormatter("%H:%M")
ax.xaxis.set_major_locator(days)
ax.xaxis.set_major_formatter(daysFmt)
fig.autofmt_xdate()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
plt.xticks(rotation=70)

leg = plt.legend(loc='best')
leg.get_frame().set_linewidth(0.0)

DATE = Time[0].date()
plt.text(0.4, 1.05,DATE, fontsize = 18, transform=ax.transAxes)

plt.grid()
ylim(200.0,400.0)
if savename:
    plt.savefig(savename)

plt.show()

