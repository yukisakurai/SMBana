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
parser.add_argument('-f', '--file',
                    nargs=1,
                    action='store',
                    type=str,
                    default='20160311_YSTM',
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
                    default=[10000000000000000000],
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
                    default=['.'],
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
now = datetime.datetime.now()
DATE = now.strftime('%Y%m%d')
if not os.path.exists('plot/' + DATE):
    os.mkdir('plot/' + DATE)
savename = 'plot/' + DATE + '/TempSensorCalib_' + now.strftime('%H%M%S') + '.png'

dir = args.file.split('_')[0] + '_' + args.file.split('_')[1]
file = dir.split('_')[0] + '_YSTM_calib'
filename = os.getcwd() + '/../data/' + dir + '/' + file + '.txt'
if os.path.exists(filename):
    print ''
    print '--------------------------------------------------'
    print 'READ FILE : %s' % file
    print ''

    ttime, ch, ch58_T, B = lib_smb.read_VTBM(filename,sampling)
    relttime = lib_smb.get_rel_time_hour(ttime)

else:
    print 'error : %s does not exist' % filename
    exit()

print ''
print '--------------------------------------------------'
print 'PLOTTING'
print ''

# Calibrate ch5
ch5_calibT = lib_temp.CalibSiPD(ch[4],'D6059358')

# Calcurate Resistance
resist = []
for i in range(4):
    resist.append(ch[i] * -1e5)


# CANVAS
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.15,right=0.95)

# ================
# PLOTS
# ================
ax = fig.add_subplot(111)
# xtitle = 'Time [hour]'
xtitle = 'Temprature [K]'
ytitle = r'Resistance [$\Omega$]'
# ytitle = 'Voltage [V]'
# ytitle = 'Magnetic Field [kG]'

plt.xlabel(xtitle, fontsize=24)
plt.ylabel(ytitle, fontsize=24)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.1, 0.5)

for i in range(3,4):
    plt.plot(ch5_calibT,
             ch[i] * -1e5,
             style,
             color=args.color[i],
             linewidth = 2.0,
             markersize = 5,
             label='ch'+str(i+1))


# polynominal fit
target_ch = ch[3]
idx1 = np.where(ch5_calibT>50)
T1 = ch5_calibT[idx1]
R1 = (target_ch * -1e5)[idx1]
x1 = np.linspace(T1.min(), T1.max(), 10000)
p1 = np.poly1d(np.polyfit(T1, R1, 10))
plt.plot(x1,
         p1(x1),
         '-',
         color=args.color[1],
         linewidth = 2.0,
         label='ch'+str(i+1) + ' fit')

idx2 = np.where((ch5_calibT<=50.2) & (ch5_calibT>20))
T2 = ch5_calibT[idx2]
R2 = (target_ch * -1e5)[idx2]
x2 = np.linspace(T2.min(), T2.max(), 10000)
p2 = np.poly1d(np.polyfit(T2, R2, 10))
plt.plot(x2,
         p2(x2),
         '-',
         color=args.color[1],
         linewidth = 2.0,
         label='ch'+str(i+1) + ' fit')

idx3 = np.where((ch5_calibT<=20.2) & (ch5_calibT>10))
T3 = ch5_calibT[idx3]
R3 = (target_ch * -1e5)[idx3]
x3 = np.linspace(T3.min(), T3.max(), 10000)
p3 = np.poly1d(np.polyfit(T3, R3, 10))
plt.plot(x3,
         p3(x3),
         '-',
         color=args.color[1],
         linewidth = 2.0,
         label='ch'+str(i+1) + ' fit')

idx4 = np.where(ch5_calibT<=10.2)
T4 = ch5_calibT[idx4]
R4 = (target_ch * -1e5)[idx4]
x4 = np.linspace(T4.min(), T4.max(), 10000)
p4 = np.poly1d(np.polyfit(T4, R4, 5))
plt.plot(x4,
         p4(x4),
         '-',
         color=args.color[1],
         linewidth = 2.0,
         label='ch'+str(i+1) + ' fit')
print p1[0]
print p1

if args.ylim:
    ylim(args.ylim[0],args.ylim[1])
if args.xlim:
    xlim(args.xlim[0],args.xlim[1])


# LEGEND
leg = plt.legend(loc='best',fontsize = 20)
leg.get_frame().set_linewidth(0.0)
for i in range(len(leg.legendHandles)):
    leg.legendHandles[i]._legmarker.set_markersize(20)


plt.grid()

# plt.xscale('log')
# plt.yscale('log')

if args.save:
    plt.savefig(savename)
plt.show()
