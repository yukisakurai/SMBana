from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import spline
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoLocator
import lib_smb as lib_smb
import argparse
import os.path

parser = argparse.ArgumentParser(description='description : This is plotting macro for driver RPM')
parser.add_argument('-f', '--file', nargs=1, type=str, help=' : input file name')
parser.add_argument('-s', '--sampling', nargs=1, type=int, help=' : number of sampling')
parser.add_argument('-c', '--channel', nargs=1, type=int, help=' : channel number')
args = parser.parse_args()

name = '20160124_YSTM_run0003'
if args.file:
    path = '../data/' + name +'.txt'
    if os.path.exists(path):
        name = args.file[0]
    else:
        print 'error: input file does not exist'
        sys.exit()

sampling =   1000
if args.sampling:
    sampling = args.sampling[0]

channel = 1
if args.channel:
    if (args.channel[0] > -1) & (args.channel[0] < 4):
        channel = args.channel[0]
    else:
        print 'error: channel %d does not exist' % args.channel[0]
        sys.exit()

fname = '../data/' + name +'.txt'
max_sample = 10000000

savename = 'plot/' + name + '.pdf'
if sampling != -1:
    savename = 'plot/' + name + '_ch_' + str(channel) + '_sampling_' + str(sampling) + '_waveform.pdf'

# read data file
time, enco, gaus, driver = lib_smb.read_data_3ch(fname,sampling)
rel_time = lib_smb.get_rel_time(time)
daysFmt = mdates.DateFormatter("%S.%f")
date = str(time[0].date())
hour = str(time[0].hour)
day = str(time[0].day)
minute = str(time[0].minute)
str_time = date + ' ' + hour + ':' + minute

if channel == 0:
    fig = plt.figure(figsize=(18, 6))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.9, wspace=0.2, hspace=0.15)

    ax1 = fig.add_subplot(131)
    xlabel("Time [sec]", fontsize=15)
    ylabel('Voltage [V]', fontsize=15)
    ax1.xaxis.set_label_coords(0.5, -0.18)
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.text(0.3, 1.05,str_time, fontsize = 15, transform=ax1.transAxes)
    plt.plot(rel_time, enco, '-b')
    plt.grid()

    ax2 = fig.add_subplot(132)
    xlabel("Time [sec]", fontsize=15)
    ylabel('Voltage [V]', fontsize=15)
    ax2.xaxis.set_label_coords(0.5, -0.18)
    ax2.yaxis.set_label_coords(-0.1, 0.5)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.text(0.3, 1.05,str_time, fontsize = 15, transform=ax2.transAxes)
    plt.plot(rel_time, gaus, '-g')
    plt.grid()

    ax3 = fig.add_subplot(133)
    xlabel("Time [sec]", fontsize=15)
    ylabel('Voltage [V]', fontsize=15)
    ax3.xaxis.set_label_coords(0.5, -0.18)
    ax3.yaxis.set_label_coords(-0.1, 0.5)
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(15)
    plt.text(0.3, 1.05,str_time, fontsize = 15, transform=ax3.transAxes)
    plt.plot(rel_time, driver, '-r')
    plt.grid()

else:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    xlabel("Time [sec]", fontsize=20)
    ylabel('Voltage [V]', fontsize=20)
    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.yaxis.set_label_coords(-0.1, 0.5)
    plt.text(0.35, 1.05, str_time, fontsize = 20, transform=ax.transAxes)


if channel == 1:
    delta_enco = enco - enco.mean()
    plt.plot(rel_time, delta_enco, 'ob')
    plt.grid()

if channel == 2:
    plt.plot(rel_time, gaus, '-g')
    plt.grid()

if channel == 3:
    plt.plot(rel_time, driver, '-r')
    plt.grid()

fig.autofmt_xdate()

plt.xticks(rotation=30)

if savename:
    plt.savefig(savename)

plt.show()

