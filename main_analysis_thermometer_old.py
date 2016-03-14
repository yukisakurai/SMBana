from matplotlib.pyplot import *
from pylab import *
import scipy.optimize as opt

temp = [4.2,77,305]
ch1 = [32961,653,126]
ch2 = [32594,648,126]
ch3 = [28327,612,124]
ch4 = [15256,470,103]
ch6 = [28257,609,122]

def logarithm_fit(x, a, b):
    return a*(x**b)

param_ch1, flag_ch1 = opt.curve_fit(logarithm_fit, temp, ch1, p0=(200000,-1.2))
param_ch2, flag_ch2 = opt.curve_fit(logarithm_fit, temp, ch2, p0=(200000,-1.2))
param_ch3, flag_ch3 = opt.curve_fit(logarithm_fit, temp, ch3, p0=(200000,-1.2))
param_ch4, flag_ch4 = opt.curve_fit(logarithm_fit, temp, ch4, p0=(85000 ,-1.2))
param_ch6, flag_ch6 = opt.curve_fit(logarithm_fit, temp, ch6, p0=(200000,-1.2))

# f = open('temprature_constant.txt','w')
# str_ch1 =  'ch1 ' + str(param_ch1[0]) + ' ' + str(param_ch1[1]) + '\n'
# str_ch2 =  'ch2 ' + str(param_ch2[0]) + ' ' + str(param_ch2[1]) + '\n'
# str_ch3 =  'ch3 ' + str(param_ch3[0]) + ' ' + str(param_ch3[1]) + '\n'
# str_ch4 =  'ch4 ' + str(param_ch4[0]) + ' ' + str(param_ch4[1]) + '\n'
# str_ch6 =  'ch6 ' + str(param_ch6[0]) + ' ' + str(param_ch6[1]) + '\n'
# f.writelines(str_ch1)
# f.writelines(str_ch2)
# f.writelines(str_ch3)
# f.writelines(str_ch4)
# f.writelines(str_ch6)

fit_ch1 = logarithm_fit(temp,param_ch1[0],param_ch1[1])
fit_ch2 = logarithm_fit(temp,param_ch2[0],param_ch2[1])
fit_ch3 = logarithm_fit(temp,param_ch3[0],param_ch3[1])
fit_ch4 = logarithm_fit(temp,param_ch4[0],param_ch4[1])
fit_ch6 = logarithm_fit(temp,param_ch6[0],param_ch6[1])

fig = figure(figsize=(10, 8))
ax = fig.add_subplot(111)
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1.0)
ax.patch.set_facecolor('white')
ax.patch.set_alpha(1.0)

xlabel(r'Temperature [K]', fontsize=20, fontname='arial')
ylabel(r'Resistance [$\Omega$]', fontsize=20, fontname='arial')
ax.xaxis.set_label_coords(0.5, -0.05)
ax.yaxis.set_label_coords(-0.08, 0.5)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)

plot(temp,ch1,'o',markersize=10,markerfacecolor='b',markeredgecolor='b',label='ch1:M1')
plot(temp,ch2,'o',markersize=10,markerfacecolor='g',markeredgecolor='g',label='ch2:Driver')
plot(temp,ch3,'o',markersize=10,markerfacecolor='r',markeredgecolor='r',label='ch3:Enc.Opt')
plot(temp,ch4,'o',markersize=10,markerfacecolor='c',markeredgecolor='c',label='ch4:HTS')
plot(temp,ch6,'o',markersize=10,markerfacecolor='y',markeredgecolor='y',label='ch6:HeatStamp')


plot(temp,fit_ch1,'--b',label='ch1:M1 (fit)')
plot(temp,fit_ch2,'--g',label='ch2:Driver (fit)')
plot(temp,fit_ch3,'--r',label='ch3:Enc.Opt. (fit)')
plot(temp,fit_ch4,'--c',label='ch4:HTS (fit)')
plot(temp,fit_ch6,'--y',label='ch6:HearStamp (fit)')

# text(0.05, 0.45, r'$y=ax^{b}$', fontsize = 20, transform=ax.transAxes)

xscale("log")
yscale("log")

rcParams['legend.numpoints'] = 1
leg = legend(loc='best')
leg.get_frame().set_linewidth(0.0)

grid()

filename = "plot/Thermometer.pdf"
savefig(filename)

show()
