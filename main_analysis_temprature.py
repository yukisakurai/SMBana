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

parser = argparse.ArgumentParser(description='description : main analysis script of SMB test for LiteBIRD HWP')
parser.add_argument('-f', '--file',
                    nargs=1,
                    action='store',
                    type=str,
                    default='20160115_YSTM',
                    help=' : directory name of input files')
parser.add_argument('--chname',
                    nargs='*',
                    action='store',
                    type=str,
                    default=['HTS','Rotor(Down)','HeatStrap1','HeatStrap2','Gripper','HeatStrap(Diode)','Rotor(Up)','Driver'],
                    help=' : plot draw style')
parser.add_argument('-c', '--color',
                    nargs='*',
                    action='store',
                    type=str,
                    default=['b','r','g','c','m','y','orange','limegreen'],
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
                    default=[10000000],
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
parser.add_argument('--save',
                    action='store',
                    type=bool,
                    default=False,
                    help=' : boolean for plot save')

args = parser.parse_args()
sampling = args.sampling[0]
style = args.style[0]
save = args.save
chname = args.chname
now = datetime.datetime.now()
DATE = now.strftime('%Y%m%d')
if not os.path.exists('plot/' + DATE):
    os.mkdir('plot/' + DATE)
savename = 'plot/' + DATE + '/temprature_' + now.strftime('%H%M%S') + '.png'


# ================
# READ INPUT FILE
# ================
dir = args.file.split('_')[0] + '_' + args.file.split('_')[1]
file = dir.split('_')[0] + '_TM'
filename = os.getcwd() + '/../data/' + dir + '/' + file + '.txt'
if os.path.exists(filename):
    print ''
    print '--------------------------------------------------'
    print 'READ FILE : %s' % file
    print ''

    ttime, ch = lib_smb.read_TM_6ch(filename,sampling)
    relttime = lib_smb.get_rel_time_hour(ttime)
    a = 137500.63
    b = -1.226
    for i in range(0,5):
        ch[i] = a * ch[i] ** b
else:
    print 'error : %s does not exist' % filename
    exit()

print ''
print '--------------------------------------------------'
print 'PLOTTING'
print ''

# CANVAS
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.15,right=0.95)

# ================
# PLOTS
# ================
ax = fig.add_subplot(111)
xtitle = 'Time [hour]'
ytitle = 'Temprature [K]'
plt.xlabel(xtitle, fontsize=24)
plt.ylabel(ytitle, fontsize=24)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.1, 0.5)

if args.ylim:
    plt.ylim(args.ylim[0],args.ylim[1])
if args.xlim:
    plt.xlim(args.xlim[0],args.xlim[1])

for i in [3]:
    plt.plot(relttime,
             ch[i],
             style,
             color=args.color[i],
             linewidth = 2.0,
             markersize = 4,
             label=chname[i])

# LEGEND
leg = plt.legend(loc='best',fontsize = 20)

leg.get_frame().set_linewidth(0.0)
for i in range(len(leg.legendHandles)):
    leg.legendHandles[i]._legmarker.set_markersize(8)

plt.grid()
if args.save:
    plt.savefig(savename)
plt.show()

