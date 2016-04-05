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
parser.add_argument('-f', '--file', nargs=1, type=str, help=' : input file')
parser.add_argument('-r', '--run', nargs=1, type=str, help=' : run number')
parser.add_argument('-c', '--color', nargs=1, type=str, help=' : line color')
parser.add_argument('-s', '--sampling', nargs=1, type=int, help=' : number of sample')
args = parser.parse_args()
name = '20160124_YSTM_run0003'
if args.file:
    name = args.file[0]

color = 'b'
if args.color:
    color = args.color[0]

sampling =   10000
if args.sampling:
    sampling = args.sampling[0]
max_sample = 10000000

fname = '../data/' + name +'.txt'
savename = 'plot/' + name + '.pdf'
if sampling != -1:
    savename = 'plot/' + name + '_sampling_' + str(sampling) + '_opt_freq.pdf'
savename = ''

# constant parameter
num_slots = 60.
gear_ratio = (3./5.) * (20./96.)

# read data file
time, enco, gauss, driver = lib_smb.read_data_3ch(fname,sampling)
delta_enco = enco - enco.mean()

# get zero line
zero = np.zeros(len(enco))

# get opt. enc. frequency
opt_freq_time, opt_freq = lib_smb.get_opt_ave_freq(num_slots, time, enco)
opt_freq_rel_time = lib_smb.get_rel_time(opt_freq_time)

# get driver frequency
driver_freq_time, driver_freq, driver_angle = lib_smb.get_driver_freq(gear_ratio, time, driver)
driver_freq_rel_time = lib_smb.get_rel_time(driver_freq_time)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
xlabel("Time [sec]", fontsize=20)
ylabel("Frequency [Hz]", fontsize=20)
ax.xaxis.set_label_coords(0.5, -0.05)
ax.yaxis.set_label_coords(-0.1, 0.5)

plt.plot(driver_freq_rel_time, driver_freq, '-', color = 'plum', linewidth = 2.0, label='driver')
plt.plot(opt_freq_rel_time, opt_freq, '-r', linewidth = 2.0, label='opt. enc.')

leg = plt.legend(loc='best',fontsize = 20)
leg.get_frame().set_linewidth(0.0)

date = str(time[0].date())
hour = str(time[0].hour)
day = str(time[0].day)
minute = str(time[0].minute)
str_time = date + ' ' + hour + ':' + minute
plt.text(0.35, 1.05,str_time, fontsize = 20, transform=ax.transAxes)

#plt.xticks(rotation=30)
plt.grid()

#ylim(-10.0,500.0)
if savename:
    plt.savefig(savename)

plt.show()
