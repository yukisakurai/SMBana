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
from datetime import datetime
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
                    default=['-'],
                    help=' : plot draw style')

args = parser.parse_args()
sampling = args.sampling[0]
style = args.style[0]
now = datetime.datetime.now()
DATE = now.strftime('%Y%m%d')
os.mkdir('plot/' + DATE)
savename = 'plot/' + DATE + '/constant_' + now.strf('%H%M%S')
print savename

# #### SET CONSTANT PARAMETERS ####
# num_slots = 60
# num_magnets = 96
# num_driver_magnets = 20
# gear_ratio = 3./5. * float(num_driver_magnets)/float(num_magnets)
# num_files = len(args.file)

# plt_opt_freq = []
# plt_opt_angle = []
# plt_opt_time = []
# plt_opt_label = []
# plt_gauss = []
# plt_gauss_time = []
# plt_gauss_label = []
# plt_driver_freq = []
# plt_driver_angle = []
# plt_driver_time = []
# plt_driver_label = []
# plt_delta_freq = []
# plt_delta_angle = []
# plt_delta_time = []

# for i in range(len(args.file)):

#     dir = args.file[i].split('_')[0] + '_' + args.file[i].split('_')[1]
#     run = args.file[i].split('_')[2]
#     file = args.file[i]

#     filename = os.getcwd() + '/../data/' + dir + '/' + file + '.txt'
#     if not os.path.exists(filename):
#         print 'error : %s does not exist' % filename
#         exit()

#     print ''
#     print '--------------------------------------------------'
#     print 'READ FILE (NUMBER %d) : %s' % (i+1 , file)
#     print ''

#     time, opt, gauss, driver = lib_smb.read_data_3ch(filename,sampling)

#     # get opt. enc. frequency
#     opt_time, opt_freq, opt_angle = lib_smb.get_opt_freq(num_slots, time, opt)


#     plt_opt_freq.append(opt_freq)
#     plt_opt_angle.append(opt_angle)
#     plt_opt_time.append(opt_time)
#     plt_opt_label.append('opt. enc.')

#     # get driver frequency
#     driver_time, driver_freq, driver_angle = lib_smb.get_driver_freq(gear_ratio, time, driver)
#     driver_angle = lib_smb.get_linear_fit_array(driver_angle,driver_time,
#                                                  opt_angle,opt_time)
#     driver_freq = lib_smb.get_average_array(driver_freq,opt_freq)

#     plt_driver_freq.append(driver_freq)
#     plt_driver_angle.append(driver_angle)
#     plt_driver_time.append(opt_time)
#     plt_driver_label.append('external motor')
#     plt_delta_freq.append(driver_freq - opt_freq)
#     plt_delta_angle.append(driver_angle - opt_angle)

# print ''
# print '--------------------------------------------------'
# print 'PLOTTING'
# print ''


# # CANVAS

# fig = plt.figure(figsize=(10, 8))
# fig.subplots_adjust(left=0.15,right=0.95)

# # ================
# # FREQUENCY PLOTS
# # ================
# ax = fig.add_subplot(111)
# xtitle = 'Time [sec]'
# ytitle = 'Frequency [Hz]'
# ymin = 0.75
# ymax = 1.25
# plt.xlabel(xtitle, fontsize=24)
# plt.ylabel(ytitle, fontsize=24)
# ax.xaxis.set_label_coords(0.5, -0.1)
# ax.yaxis.set_label_coords(-0.1, 0.5)
# ylim(ymin,ymax)

# for i in range(len(plt_driver_freq)):
#     plt.plot(plt_driver_time[i],
#              plt_driver_freq[i],
#              style,
#              color=args.color[i+1],
#              linewidth = 2.0,
#              markersize = 4,
#              label=plt_driver_label[i])

# for i in range(len(plt_opt_freq)):
#     plt.plot(plt_opt_time[i],
#              plt_opt_freq[i],
#              style,
#              color=args.color[i],
#              linewidth = 2.0,
#              markersize = 4,
#              label=plt_opt_label[i])


# # LEGEND
# leg = plt.legend(loc='best',fontsize=24)
# leg.get_frame().set_linewidth(0.0)
# plt.grid()


# # ===========
# # ANGLE PLOTS
# # ===========
# ax = fig.add_subplot(111)
# xtitle = 'Time [sec]'
# ytitle = r'$\theta$ [deg]'
# ymin = -10.0
# ymax = 500.0
# xlabel(xtitle, fontsize=18)
# ylabel(ytitle, fontsize=18)
# ax.xaxis.set_label_coords(0.5, -0.05)
# ax.yaxis.set_label_coords(-0.1, 0.5)
# ylim(ymin,ymax)

# for i in range(len(plt_driver_angle)):
#     plt.plot(plt_driver_time[i],
#              plt_driver_angle[i],
#              style,
#              color=args.color[i+1],
#              lw = 0.5,
#              markersize = 4,
#              label=plt_driver_label[i])

# for i in range(len(plt_opt_angle)):
#     plt.plot(plt_opt_time[i],
#              plt_opt_angle[i],
#              style,
#              color=args.color[i],
#              lw = 0.5,
#              markersize = 4,
#              label=plt_opt_label[i])

# # LEGEND
# leg = plt.legend(loc='best',fontsize = 18)
# leg.get_frame().set_linewidth(0.0)
# plt.grid()

# # =================
# # DELTA FREQUENCY HIST
# # =================
# ax = fig.add_subplot(111)
# xtitle = r'$\Delta$ Frequency [Hz]'
# ytitle = 'Entries'
# xmin = -0.15
# xmax = 0.15
# xlabel(xtitle, fontsize=18)
# ylabel(ytitle, fontsize=18)
# ax.xaxis.set_label_coords(0.5, -0.05)
# ax.yaxis.set_label_coords(-0.1, 0.5)

# for i in range(len(plt_delta_freq)):

#     hist = plt.hist(plt_delta_freq[i],
#                     bins=50,
#                     range=(xmin,xmax),
#                     normed=False,
#                     histtype='step',
#                     lw=2.0,
#                     color=args.color[i],
#                     alpha=0.5,
#                     label='data')


#     # Gaussian Fit
#     x = [0.5 * (hist[1][i] + hist[1][i+1]) for i in xrange(len(hist[1])-1)]
#     y = hist[0]
#     popt, pcov = optimize.curve_fit(lib_smb.gaussian, x, y)
#     x_fit = linspace(x[0], x[-1], 100)
#     y_fit = lib_smb.gaussian(x_fit, *popt)
#     print 'Gaussian Fit Result %i ======' % i
#     print 'norm = %f , mu = %f , sigma = %f' % ( round(popt[0],3), round(popt[1],3), round(popt[2],3) )
#     print '=========================='
#     plt.plot(x_fit, y_fit, '--', color='r', lw=2, label='fit')

# ymin = 0.0
# ymax = hist[0].max() * 1.4
# ylim(ymin,ymax)
# xmin = -0.1
# xmax = 0.1
# xlim(xmin,xmax)
# plt.grid()

# # LEGEND
# leg = plt.legend(loc='best',fontsize=18)
# leg.get_frame().set_linewidth(0.0)

# # =================
# # DELTA ANGLE HIST
# # =================
# ax = fig.add_subplot(111)
# xtitle = r'$\Delta$ $\theta$ [deg]'
# ytitle = 'Entries'
# xmin = plt_delta_angle[0].min()
# xmax = plt_delta_angle[0].max()
# xlabel(xtitle, fontsize=18)
# ylabel(ytitle, fontsize=18)
# ax.xaxis.set_label_coords(0.5, -0.05_)
# ax.yaxis.set_label_coords(-0.1, 0.5)

# for i in range(len(plt_delta_angle)):
#     hist = plt.hist(plt_delta_angle[i],
#                     bins=25,
#                     range=(xmin,xmax),
#                     normed=False,
#                     histtype='step',
#                     lw=2.0,
#                     color=args.color[i],
#                     alpha=0.5,
#                     label='data')

#     # Gaussian Fit
#     x = [0.5 * (hist[1][i] + hist[1][i+1]) for i in xrange(len(hist[1])-1)]
#     y = hist[0]
#     popt, pcov = optimize.curve_fit(lib_smb.gaussian, x, y)
#     x_fit = linspace(x[0], x[-1], 100)
#     y_fit = lib_smb.gaussian(x_fit, *popt)
#     print 'Gaussian Fit Result %i ======' % i
#     print 'norm = %f , mu = %f , sigma = %f' % ( round(popt[0],3), round(popt[1],3), round(popt[2],3) )
#     print '=========================='
#     plt.plot(x_fit, y_fit, '--', color='r', lw=2, label='fit')


# ymin = 0.0
# ymax = hist[0].max() * 1.4
# ylim(ymin,ymax)
# xmin = -2.5
# xmax = 2.5
# xlim(xmin,xmax)

# # LEGEND
# leg = plt.legend(loc='best',fontsize = 18)
# leg.get_frame().set_linewidth(0.0)
# plt.grid()

# # ===========
# # FREQUENCT HIST
# # ===========
# ax = fig.add_subplot(111)
# xtitle = 'Frequency [Hz]'
# ytitle = 'Entries'
# xmin = 0.9
# xmax = 1.1
# xlabel(xtitle, fontsize=24)
# ylabel(ytitle, fontsize=24)
# ax.xaxis.set_label_coords(0.5, -0.08)
# ax.yaxis.set_label_coords(-0.1, 0.5)

# for i in range(len(plt_driver_freq)):

#     hist_driver = plt.hist(plt_driver_freq[i],
#                            bins=25,
#                            range=(xmin,xmax),
#                            normed=False,
#                            histtype='step',
#                            lw=2.0,
#                            color=args.color[i+1],
#                            alpha=0.5,
#                            label='external motor')

#     # Gaussian Fit
#     x = [0.5 * (hist_driver[1][j] + hist_driver[1][j+1]) for j in xrange(len(hist_driver[1])-1)]
#     y = hist_driver[0]
#     init_param = [len(plt_driver_freq[i]), 1.0, 0.1]
#     popt, pcov = optimize.curve_fit(lib_smb.gaussian, x, y, init_param)
#     x_fit = linspace(x[0], x[-1], 100)
#     y_fit = lib_smb.gaussian(x_fit, *popt)
#     print ''
#     print 'Gaussian Fit Result %i ======' % i
#     print 'norm = %f , mu = %f , sigma = %f' % ( round(popt[0],3), round(popt[1],3), round(popt[2],3) )
#     print '=========================='
#     print ''
#     fit_driver = plt.plot(x_fit, y_fit, '--', color='r', lw=2, label='external motor (fit)')

# for i in range(len(plt_opt_freq)):
#     hist_opt = plt.hist(plt_opt_freq[i],
#                         bins=25,
#                         range=(xmin,xmax),
#                         normed=False,
#                         histtype='step',
#                         lw=2.0,
#                         color=args.color[i],
#                         alpha=0.5,
#                         label='opt. enc.')

#     # Gaussian Fit
#     x = [0.5 * (hist_opt[1][j] + hist_opt[1][j+1]) for j in xrange(len(hist_opt[1])-1)]
#     y = hist_opt[0]
#     init_param = [len(plt_opt_freq[0]), 1.0, 0.05]
#     popt, pcov = optimize.curve_fit(lib_smb.gaussian, x, y, init_param)
#     x_fit = linspace(x[0], x[-1], 100)
#     y_fit = lib_smb.gaussian(x_fit, *popt)
#     print ''
#     print 'Gaussian Fit Result %i ======' % i
#     print 'norm = %f , mu = %f , sigma = %f' % ( round(popt[0],3), round(popt[1],3), round(popt[2],3) )
#     print '=========================='
#     print ''
#     fit_opt = plt.plot(x_fit, y_fit, '--', color='b', lw=2, label='opt. enc. (fit)')

# ymin = 0.0
# ymax = hist_driver[0].max() * 1.2
# ylim(ymin,ymax)


# # LEGEND
# leg = plt.legend(loc='upper left',fontsize = 20)
# leg.get_frame().set_linewidth(0.0)
# plt.grid()

