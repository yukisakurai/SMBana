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
import lib_temp as lib_temp

parser = argparse.ArgumentParser(description='description : main analysis script of SMB test for LiteBIRD HWP')
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
parser.add_argument('--save',
                    action='store',
                    type=bool,
                    default=False,
                    help=' : boolean for plot save')

args = parser.parse_args()
style = args.style[0]
save = args.save
now = datetime.datetime.now()
DATE = now.strftime('%Y%m%d')
if not os.path.exists('plot/' + DATE):
    os.mkdir('plot/' + DATE)
savename = 'plot/' + DATE + '/CalibCurve_' + now.strftime('%H%M%S') + '.png'

volt = linspace(0.505, 1.65, 10000)
temp = lib_temp.CalibSiPD(volt,'D6059358')

# CANVAS
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.15,right=0.95)

# ================
# PLOTS
# ================
ax = fig.add_subplot(111)
xtitle = 'Temprature [K]'
ytitle = 'Voltage [V]'
plt.xlabel(xtitle, fontsize=24)
plt.ylabel(ytitle, fontsize=24)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.1, 0.5)

if args.ylim:
    ylim(args.ylim[0],args.ylim[1])
if args.xlim:
    xlim(args.xlim[0],args.xlim[1])

plt.plot(temp,
         volt,
         style,
         color=args.color[0],
         linewidth = 1.0,
         markersize = 1,
         label='Curve')


plt.grid()
if args.save:
    plt.savefig(savename)

plt.show()
