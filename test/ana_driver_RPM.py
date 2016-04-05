from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import spline
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoLocator
import lib_smb as lib_smb
import argparse

parser = argparse.ArgumentParser(description='description : This is plotting macro for driver RPM')
parser.add_argument('-f', '--file', nargs=1, type=str, help=' : input file name')
parser.add_argument('-r', '--input_rpm', nargs=1, type=int, help=' : input RPM value')
parser.add_argument('-c', '--color', nargs=1, type=str, help=' : line color')
parser.add_argument('-s', '--sampling', nargs=1, type=int, help=' : number of sample')
args = parser.parse_args()

name = '20160124_YSTM_run0003'
if args.file:
    name = args.file[0]

input_RPM = 244
if args.input_rpm:
    input_RPM = args.input_rpm[0]

color = 'b'
if args.color:
    color = args.color[0]

fname = '../data/' + name +'.txt'

sampling =   1000
if args.sampling:
    sampling = args.sampling[0]
max_sample = 10000000

savename = 'plot/' + name + '.pdf'
if sampling != -1:
    savename = 'plot/' + name + '_RPM_' + str(input_RPM) + '.pdf'

#savename = ''

# read data file
array_time, array_enco, array_gaus, array_driver = lib_smb.read_data_3ch(fname,sampling)
delta_array_driver = array_driver - array_driver.mean()

# get RPM
array_RPM_time, array_RPM = lib_smb.get_array_driver_RPM(array_time, array_driver)

# get input RPM line
list_input_RPM = []
for i in range(0,len(array_RPM_time)):
    list_input_RPM.append(input_RPM)
array_input_RPM = np.array(list_input_RPM)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
xlabel("Time [sec]", fontsize=20)
ylabel("Driver RPM [r/min]", fontsize=20)
ax.xaxis.set_label_coords(0.5, -0.18)
ax.yaxis.set_label_coords(-0.1, 0.5)

plt.plot(array_RPM_time, array_RPM, '-' + color, linewidth = 1.0, label='Speed Out Signal')
plt.plot(array_RPM_time, array_input_RPM, '--' + color, linewidth = 1.0, label='Input RPM = ' + str(input_RPM))

ax.xaxis.set_major_locator(AutoLocator())
daysFmt = mdates.DateFormatter("%S.%f")
ax.xaxis.set_major_formatter(daysFmt)
fig.autofmt_xdate()

date = str(array_time[0].date())
hour = str(array_time[0].hour)
day = str(array_time[0].day)
minute = str(array_time[0].minute)
str_time = date + ' ' + hour + ':' + minute
plt.text(0.35, 1.05,str_time, fontsize = 20, transform=ax.transAxes)

plt.xticks(rotation=30)
plt.grid()

leg = plt.legend(loc='best',fontsize = 20)
leg.get_frame().set_linewidth(0.0)

ylim(input_RPM - 150.0,input_RPM + 150.0)
if savename:
    plt.savefig(savename)

plt.show()
