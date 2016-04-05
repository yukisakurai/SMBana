from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import spline
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoLocator
import lib_smb as lib_smb

# n = 0 ~ 4 : RPM = 100 ~ 500

for n in range(3,5):
    # driver RPM 100 -> 500 (100 step)
    list_freq = []
    list_freq_driver = []

    # constant parameter
    num_slots = 60.
#    list_driver_RPM = [100,210,324,405,503]
    list_driver_RPM = [503,400,303,210,106]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    xlabel("Time", fontsize=20)
    ylabel("Frequency [Hz]", fontsize=20)
    ax.xaxis.set_label_coords(0.5, -0.12)
    ax.yaxis.set_label_coords(-0.1, 0.5)

    run_num = 37 + n
    name = '20160111_YSTM_run0' + str(run_num)
    if n >= 3:
        run_num = n - 1
        name = '20160114_YSTM_run00' + str(run_num)

    fname = '../data/' + name +'.txt'
    sampling =   1000000
#    sampling = 10000
    max_sample = 10000000

    driver_RPM = list_driver_RPM[n]
    driver = (driver_RPM/60.) * (3./5.)

    savename = 'plot/' + name + '_driver_RPM_' + str(500-n*100) + '.pdf'

    # read data file
    array_time, array_enco, array_gaus = lib_smb.read_data_2ch(fname,sampling)
    delta_array_enco = array_enco - array_enco.mean()

    # get frequency
    array_freq_time, array_freq = lib_smb.get_array_freq(num_slots,array_time, array_enco)

    # get driver frequency
    list_driver = []
    for i in range(0,len(array_freq_time)):
        list_driver.append(driver)
        array_driver = np.array(list_driver)


    label_hwp = 'HWP (r/min=' + str(driver_RPM) + ')'
    label_driver = 'Driver (r/min=' + str(driver_RPM) + ')'
    # color = ['k','r','b','g','y']
    color = ['y','g','b','r','k']
    plt.plot(array_freq_time, array_freq, '-' + color[n], linewidth = 4.0, label=label_hwp)
#    plt.plot(array_freq_time, array_driver, '--' + color[n], linewidth = 4.0, label=label_driver)

    ax.xaxis.set_major_locator(AutoLocator())
    daysFmt = mdates.DateFormatter("%H:%M:%S")
    ax.xaxis.set_major_formatter(daysFmt)
    fig.autofmt_xdate()

    plt.xticks(rotation=30)
    plt.grid()

    leg = plt.legend(loc='best')
    leg.get_frame().set_linewidth(0.0)

    Text = 'Driver r/min = ' + str(driver_RPM)
    plt.text(0.4, 1.05,Text, fontsize = 18, transform=ax.transAxes)

    ylim(0.0,1.0)
    if savename:
        plt.savefig(savename)

plt.show()
