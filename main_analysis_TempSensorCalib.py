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

ch_serial = ['x74322', 'x74376', 'x74355', 'x74391', 'D6059358', 'D6032612', 'D6027096', 'D6025126']
# polynominal fit
for i in range(0,4):
    target_ch = ch[i]
    THR = np.array([])
    PARAM = np.array([])
    param = []
    threshold = []
    for j in range(0,4):
        T = ch5_calibT
        R = (target_ch * -1e5)
        if j == 0:
            idx = np.where(R<1000)
            npol = 10
            t = 1000
        if j == 1:
            idx = np.where((R<=3000) & (R>1000))
            npol = 10
            t = 1000
        if j == 2:
            idx = np.where((R<=7000) & (R>3000))
            npol = 10
            t = 3000
        if j == 3:
            idx = np.where(R>7000)
            npol = 5
            t = 7000

        T = T[idx]
        R = R[idx]
        p = np.poly1d(np.polyfit(R, T, npol))
        param.append(p)
        threshold.append(t)
    THR = np.append(THR,threshold)
    PARAM = np.append(PARAM,param)
    np.savez(os.path.join('CalibData/', ch_serial[i]),THR=THR,PARAM=PARAM)


# CANVAS
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.15,right=0.95)

# ================
# PLOTS
# ================
ax = fig.add_subplot(111)
ytitle = 'Temperature [K]'
xtitle = r'Resistance [$\Omega$]'
# xtitle = 'Time [hour]'
# ytitle = 'Voltage [V]'
# ytitle = 'Magnetic Field [kG]'

plt.xlabel(xtitle, fontsize=24)
plt.ylabel(ytitle, fontsize=24)
ax.xaxis.set_label_coords(0.5, -0.1)
ax.yaxis.set_label_coords(-0.1, 0.5)

T = ch5_calibT
R = ch[0] * -1e5
plt.plot(R,
         T,
         style,
         color=args.color[0],
         linewidth = 2.0,
         markersize = 5,
         label='ch'+str(1))

param_x74322 = np.load('CalibData/x74322.npz')['PARAM']

for i in range(4):
    p = param_x74322[i]
    if i == 0: x = np.linspace(R.min(),1000,100)
    if i == 1: x = np.linspace(1000,3000,100)
    if i == 2: x = np.linspace(3000,7000,100)
    if i == 3: x = np.linspace(7000,R.max(),100)
    plt.plot(x,
             np.polyval(p,x),
             '-',
             color='r',
             linewidth = 2.0)


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

plt.xscale('log')
plt.yscale('log')

if args.save:
    plt.savefig(savename)
plt.show()
