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

parser = argparse.ArgumentParser(description='description : main analysis script of SMB test for LiteBIRD HWP')
parser.add_argument('-f', '--file',
                    nargs='*',
                    action='store',
                    type=str,
                    default=['20160202_YSTM_run001'],
                    help=' : directory name of input files')
parser.add_argument('-c', '--color',
                    nargs='*',
                    action='store',
                    type=str,
                    default=['b','r','g','c','m','y','k'],
                    help=' : line color')
parser.add_argument('--subcolor',
                    nargs='*',
                    action='store',
                    type=str,
                    default=['grey','lightblue','plum','lightgreen'],
                    help=' : line color')
parser.add_argument('-s', '--sampling',
                    nargs=1,
                    action='store',
                    type=int,
                    default=[10000],
                    help=' : number of sample')
parser.add_argument('--xlim',
                    nargs=2,
                    action='store',
                    type=float,
                    default=None,
                    help=' : limit range of x axis')
parser.add_argument('--ylim',
                    nargs=2,
                    action='store',
                    type=float,
                    default=None,
                    help=' : limit range of y axis')
parser.add_argument('--style',
                    nargs=1,
                    action='store',
                    type=str,
                    default=['o'],
                    help=' : plot draw style')

args = parser.parse_args()
sampling = args.sampling[0]
style = args.style[0]

#### SET CONSTANT PARAMETERS ####
num_slots = 60
num_magnets = 96
num_driver_magnets = 20
gear_ratio = 3./5. * float(num_driver_magnets)/float(num_magnets)
num_files = len(args.file)

plt_opt = []
plt_opt_time = []
plt_opt_label = []
plt_opt_start_point = []
plt_opt_start_time = []

plt_gauss = []
plt_gauss_time = []
plt_gauss_label = []
plt_driver = []
plt_driver_time = []
plt_driver_label = []

for i in range(len(args.file)):

    dir = args.file[i].split('_')[0] + '_' + args.file[i].split('_')[1]
    run = args.file[i].split('_')[2]
    file = args.file[i]

    filename = os.getcwd() + '/../data/' + dir + '/' + file + '.txt'
    if not os.path.exists(filename):
        print 'error : %s does not exist' % filename
        exit()

    print ''
    print '--------------------------------------------------'
    print 'READ FILE (NUMBER %d) : %s' % (i+1 , file)
    print ''

    time, opt, gauss, driver = lib_smb.read_data_3ch(filename,sampling)
    rel_time = lib_smb.get_rel_time(time)

    # get opt. enc. frequency
    opt_time, opt_freq, opt_angle = lib_smb.get_opt_freq(num_slots, time, opt)

    delta_opt = opt - opt.mean()
    plt_opt.append(delta_opt)
    plt_opt_time.append(rel_time)
    plt_opt_label.append('opt. enc.')

    opt_start_time, opt_start_point = lib_smb.get_opt_start_point(time, opt)
    plt_opt_start_point.append(opt_start_point)
    plt_opt_start_time.append(opt_start_time)

    # get driver frequency
#     driver_time, driver_freq, driver_angle = lib_smb.get_driver_freq(gear_ratio, time, driver)
#     driver_time, driver_RPM = lib_smb.get_driver_RPM(time, driver)
#     driver_angle = lib_smb.get_linear_fit_array(driver_angle,driver_time,
#                                                 opt_angle,opt_time)
#     driver_freq = lib_smb.get_average_array(driver_freq,opt_freq)

#     print driver_RPM
#     print driver_freq
#     print driver_angle

    plt_driver.append(driver)
    plt_driver_time.append(rel_time)
    plt_driver_label.append('driver')

print ''
print '--------------------------------------------------'
print 'PLOTTING'
print ''


# CANVAS

fig = plt.figure(figsize=(16, 10))
fig.subplots_adjust(left=0.07,right=0.95)
# ================
# FREQUENCY PLOTS
# ================
ax = fig.add_subplot(111)
xtitle = 'Time [sec]'
ytitle = 'Voltage [V]'
ymin = plt_opt[0].min()
ymax = plt_opt[0].max() * 1.4
xlabel(xtitle, fontsize=14)
ylabel(ytitle, fontsize=14)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.1, 0.5)
ylim(ymin,ymax)

for i in range(len(plt_opt)):
    plt.plot(plt_opt_time[i],
             plt_opt[i],
             style,
             color=args.color[i],
             linewidth = 2.0,
             markersize = 4,
             label=plt_opt_label[i])

for i in range(len(plt_opt_start_point)):
    plt.plot(plt_opt_start_time[i],
             plt_opt_start_point[i],
             style,
             color=args.color[i+1],
             linewidth = 2.0,
             markersize = 4,
             label=plt_opt_label[i])



# LEGEND
leg = plt.legend(loc='best',fontsize = 14)
leg.get_frame().set_linewidth(0.0)
plt.grid()

plt.show()


