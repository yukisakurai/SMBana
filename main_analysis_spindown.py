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
                    default=['20160203_YSTM_run002'],
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
plt_opt_delta = []
plt_opt_time = []
plt_opt_label = []
plt_gauss = []
plt_gauss_time = []
plt_gauss_label = []

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

    # get opt. enc. frequency
    opt_time, opt_freq, opt_angle = lib_smb.get_opt_freq(num_slots, time, opt)
    opt_time_period, opt_freq_period, opt_delta_freq = \
        lib_smb.get_opt_freq_period(num_slots, time, opt)

    plt_opt.append(opt_freq_period)
    plt_opt_delta.append(opt_delta_freq)
    plt_opt_time.append(opt_time_period)
    plt_opt_label.append('opt. enc.')


print ''
print '--------------------------------------------------'
print 'PLOTTING'
print ''


# # CANVAS
ymin = 0.01
ymax = 100.0
# xtitle = 'Time [sec]'
# ytitle = 'Frequency [Hz]'
xtitle = 'f [Hz]'
ytitle = r'$\Delta$ f / f'
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
xlabel(xtitle, fontsize=24)
ylabel(ytitle, fontsize=24)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.1, 0.5)

# ===========
# PLOTS
# ===========

for i in range(len(plt_opt)):
#     plt.plot(plt_opt_time[i],
#              plt_opt[i],
#              style,
#              color=args.color[i],
#              markersize = 3,
#              lw = 1,
#              label=plt_opt_label[i])

#     # Exponential fit =====
#     init_param = [1.0, 1.0e-04, 0.1]
#     # fit range
#     fit_min = 0.0
#     fit_max = 2000.0
#     opt_fit = np.array([])
#     opt_time_fit = np.array([])
#     for j in range(len(plt_opt_time[i])):
#         if (plt_opt_time[i][j] > fit_min) and (plt_opt_time[i][j] < fit_max) :
#             opt_fit = np.append(opt_fit,plt_opt[i][j])
#             opt_time_fit = np.append(opt_time_fit,plt_opt_time[i][j])
#     popt, pcov = optimize.curve_fit(lib_smb.exponential, opt_time_fit, opt_fit, init_param)
#     x = linspace(plt_opt_time[i][0],plt_opt_time[i][-1],100)
#     y = lib_smb.exponential(x, *popt)
#     plt.plot(x, y, '--', color='r', lw=3, label='fit')

    plt.plot(plt_opt[i],
             plt_opt_delta[i],
             style,
             color=args.color[i],
             lw = 1,
             markersize=3.0,
             label=plt_opt_label[i])

    # Spindown fit =====
    init_param = [1.0, 1.0e-04, 0.1]
    # fit range
    fit_min = 0.1
    fit_max = 0.9
    opt_y_fit = np.array([])
    opt_x_fit = np.array([])
    for j in range(len(plt_opt[i])):
        if (plt_opt[i][j] > fit_min) and (plt_opt[i][j] < fit_max) :
            opt_y_fit = np.append(opt_y_fit,plt_opt_delta[i][j])
            opt_x_fit = np.append(opt_x_fit,plt_opt[i][j])
    popt, pcov = optimize.curve_fit(lib_smb.spindown, opt_x_fit, opt_y_fit)
    x = linspace(0.0,1.0,100)
    y = lib_smb.spindown(x, *popt)
    plt.plot(x, y, '--', color='r', lw=3, label='fit')

    perr = perr = np.sqrt(np.diag(pcov))
    print perr
    print 'Spindown fit result ======'
    print 'c0 = %f , c1 = %f' % (popt[0],popt[1])
    print 'c0_err = %f , c1_err = %f' % (perr[0],perr[1])



# LEGEND
leg = plt.legend(loc='best',fontsize = 24)
leg.get_frame().set_linewidth(0.0)

ylim(ymin,ymax)
if args.ylim:
    ylim(args.ylim[0],args.ylim[1])
if args.xlim:
    xlim(args.xlim[0],args.xlim[1])


yscale('log')
plt.grid()
plt.show()
