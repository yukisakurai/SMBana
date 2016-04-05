import numpy as np
import pylab as py
import lib_smb as lib_s
import lib_m as lib_m
from scipy.optimize import curve_fit
import sys
import os

'''
main_analyze_rot.py
Written by T. Matsumura, 2015-4-5

'''


# DEFINE THE BASIC PARAMETERS
pi = np.pi
num_slots = 60.


# READ THE DATA
print ''
print 'READ THE DATA'

dir_data = sys.argv[1]
dir_out = sys.argv[2]
filename = sys.argv[3]
data_enco, data_hall = lib_m.read_txt2f_skipn(dir_data+'/'+filename+'.txt',1)
#data_enco, data_hall = lib_m.read_txt2f_cvs_skipn(dir_data+'/'+filename+'.csv',1)

num = len(data_enco)
samplerate = float(sys.argv[4])
d_time = 1./ samplerate
time = np.arange(num)*d_time
print ''
print dir_data
print dir_out
print filename


# ANALYZE THE OPTICAL ENCODER DATA
print ''
print 'ANALYZE THE OPTICAL ENCODER DATA'

del_data_enco = data_enco - np.mean(data_enco)
time_flag = []
for i in range(1,num):
	if ((del_data_enco[i]>0) & (del_data_enco[i-1]<0)):
		time_interp = -(time[i]*del_data_enco[i-1]-time[i-1]*del_data_enco[i])/(del_data_enco[i]-del_data_enco[i-1])
		time_flag.append(time_interp)
#	if ((del_data_enco[i]<0) & (del_data_enco[i-1]>0)): flag[i] = -1
num_flag = len(time_flag)
rot_freq = np.zeros((2,num_flag-1))
angle = np.zeros((3,num_flag-1))
angle_i = 0.; angle_wrap_i = 0.
for i in range(num_flag-1):
	rot_freq[0,i] = 0.5*(time_flag[i+1]+time_flag[i])
	rot_freq[1,i] = 1./(time_flag[i+1]-time_flag[i])/float(num_slots)
	angle_i += 6.
	angle_wrap_i += 6.
	angle[0,i] += angle_i
	angle[1,i] += angle_wrap_i
	if (angle_i%360) == 0:
		angle_wrap_i = 0.
		angle[2,i] = 1

py.figure(1)
py.subplot(311)
py.plot(time,data_hall)
py.xlabel('Time [sec]')
py.ylabel('Hall sensor [V]')
py.title(filename)
py.subplot(312)
py.plot(time,del_data_enco)
py.xlabel('Time [sec]')
py.ylabel('Opt. Enc. out [V]')
py.subplot(313)
py.plot(rot_freq[0],rot_freq[1],'.')
py.xlabel('Time [sec]')
py.ylabel('Rot. Freq. [Hz]')
print dir_out+'/data_overview.png'
py.savefig(dir_out+'/data_overview.png')
py.clf()

# CALCULATE THE PSD OF THE HALL SENSOR DATA
PSD = lib_m.calPSD(data_hall-np.mean(data_hall), samplerate, 3)

py.figure(0,figsize=(6,9))
py.subplot(311)
py.title(filename)
py.plot(time,data_hall)
#py.ylim([1.e-13,1.e-6])
py.xlabel('Time [sec]')
py.ylabel('Hall sensor [kGauss]')

py.subplot(312)
py.plot(PSD[0][1::],PSD[1][1::])
py.semilogy()
#py.ylim([1.e-13,1.e-6])
py.xlabel('Frequency [Hz]')
py.ylabel('Power Spec. [V/rtHz]')

py.subplot(313)
py.plot(PSD[0][1::],PSD[1][1::])
#py.ylim([1.e-13,1.e-6])
py.loglog()
py.xlabel('Frequency [Hz]')
py.ylabel('Power Spec. [V/rtHz]')
#py.show()
py.savefig(dir_out+'/Hall_PSD.png')
py.clf()


np.savez(dir_out+'/optenc_data_fit', \
	time=rot_freq[0],freq=rot_freq[1],\
	angle=angle, \
	psd=PSD, help='time,freq,angle,psd')



# FIT TO THE SPIN DOWN RATE
par_guess_exp = np.array([0.55,0.,1./400.])
popt_exp, pcov_exp = curve_fit(lib_s.model_exp_spin_down, rot_freq[0],rot_freq[1], p0=par_guess_exp)
print 'Fit results from exp only'
print popt_exp
print pcov_exp

f0_iguess = 4.25
a0_iguess = 0.
a1_iguess = 0.0001
t0_iguess = 0.
c_iguess = a1_iguess
d_iguess = 1./(-2.*pi)*c_iguess
b_iguess = t0_iguess
a_iguess = f0_iguess - d_iguess
par_guess_explin = np.array([a_iguess,b_iguess,c_iguess,d_iguess])
popt_explin, pcov_explin = curve_fit(lib_s.model_explin_spin_down_t0, rot_freq[0],rot_freq[1], p0=par_guess_explin)
#par_guess_explin = np.array([a_iguess,c_iguess,d_iguess])
#popt_explin, pcov_explin = curve_fit(lib_s.model_explin_spin_down, rot_freq[0],rot_freq[1], p0=par_guess_explin)
print 'Fit results from exp + const'
print popt_explin
print pcov_explin
a1 = popt_explin[2]
a0 = -2.*pi*popt_explin[3]*popt_explin[2]
f0 = popt_explin[0] + popt_explin[3]
print 't0:', popt_explin[1]
print 'a1:', a1
print 'a0:', a0
print 'f0:', f0
freq_model = lib_s.model_explin_spin_down_t0(rot_freq[0],\
	popt_explin[0],popt_explin[1],popt_explin[2],popt_explin[3])
#a1 = popt_explin[1]
#a0 = -2.*pi*popt_explin[2]*popt_explin[1]
#f0 = popt_explin[0] + popt_explin[2]
#print 'a1:', a1
#print 'a0:', a0
#print 'f0:', f0
#freq_model = lib_s.model_explin_spin_down(rot_freq[0],popt_explin[0],popt_explin[1],popt_explin[2])

# OUTPUT THE PLOTS

py.figure(2)
py.subplot(211)
py.title(filename)
py.plot(time,del_data_enco)
py.xlabel('Time [sec]')
py.ylabel('Opt. Enc. out [V]')

py.subplot(212)
py.plot(rot_freq[0],rot_freq[1],'.')
py.plot(rot_freq[0],freq_model,'r-')
py.xlabel('Time [sec]')
py.ylabel('Rot. Freq. [Hz]')

py.savefig(dir_out+'/optenc_fit.png')
py.clf()

py.figure(4)
py.subplot(211)
py.title(filename)
py.plot(time,del_data_enco)
py.xlabel('Time [sec]')
py.ylabel('Opt. Enc. out [V]')

py.subplot(212)
py.plot(rot_freq[0],rot_freq[1],'.')
py.plot(rot_freq[0],freq_model,'r-')
py.xlabel('Time [sec]')
py.ylabel('Rot. Freq. [Hz]')
py.ylim([0,1])

py.savefig(dir_out+'/optenc_fit_zoom.png')
py.clf()

py.figure(3)
py.subplot(211)
py.plot(rot_freq[0],angle[0],'-')
py.ylabel('Angle [degs]')
py.xlabel('Time [sec]')
py.subplot(212)
py.plot(rot_freq[0],angle[1],'-')
py.ylabel('Wrap Angle [degs]')
py.xlabel('Time [sec]')
py.savefig(dir_out+'/optenc_travel.png')
py.clf()


np.savez(dir_out+'/optenc_data_fit', \
	time=rot_freq[0],freq=rot_freq[1],\
	par=popt_explin, par_err=pcov_explin, angle=angle, \
	psd=PSD, help='time,freq,par,par_err,angle,psd')


sys.exit()

num_fit = 100
freq_fit = np.arange(num_fit)/float(num_fit)*max(rot_freq[0])
#fit_exp = model_exp_spin_down(freq_fit,popt_exp[0],popt_exp[1],popt_exp[2])
fit_explin = lib_s.model_explin_spin_down(freq_fit,popt_explin[0],popt_explin[1],popt_explin[2],popt_explin[3])
#py.plot(freq_fit,fit_exp)

py.plot(freq_fit,fit_explin,'r')
py.ylabel('Rot. rate [Hz]')
py.xlabel('Time [sec]')
py.ylim([0, f0*1.2 ])


py.subplot(222)
popt_lin, pcov_lin = curve_fit(lib_s.model_lin, rot_freq[0],angle[0])
print popt_lin
print pcov_lin
py.plot(rot_freq[0],angle[0],'.')
x_ = np.arange(num_fit)/float(num_fit)*max(rot_freq[0])
py.plot(x_, lib_s.model_lin(x_,popt_lin[0],popt_lin[1]),'r-')
py.ylabel('Rot. angle [degs]')

py.subplot(223)
py.plot(rot_freq[0],angle[1],'.')


py.subplot(224)
py.plot(rot_freq[0], angle[0] - lib_s.model_lin(rot_freq[0],popt_lin[0],popt_lin[1]) ,'.')
py.ylabel('del. Rot. angle [degs]')

py.savefig(dir_out+'/optenc_travel.png')
py.clf()

py.figure(4)
py.subplot(211)
py.plot(rot_freq[0],angle[1],'-')
py.xlabel('Time [sec]')
py.ylabel('Wrap. angle [degs]')
py.subplot(212)
py.plot(rot_freq[0],angle[1],'-')
py.ylabel('Wrap. angle [degs]')
py.xlabel('Time [sec]')
py.savefig(dir_out+'/optenc_wraptravel.png')
py.clf()

sys.exit()



# ANALYZE THE HALL SENSOR DATA
del_data_hall = data_hall-np.mean(data_hall)
filtertype = 'cosine_lowpass'
par = [4.,2]
freq_, psd_hall, psd_hall_filt, data_hall_filt, filt = lib_m.AppFilt(del_data_hall, samplerate, 6, filtertype, par)
filtertype = 'cosine_lowpass'
par = [4.,2]
freq_, psd_hall, psd_hall_filt, data_hall_filt, filt = lib_m.AppFilt(del_data_hall, samplerate, 6, filtertype, par)


num1 = 10000
num2 = 12000

py.figure(1)
py.subplot(211)
#py.plot(time,del_data_hall)
py.plot(rot_freq[0],angle[1],'-')
py.xlim([10,20])
py.xlabel('Time [sec]')
py.subplot(212)
#py.plot(time,flag*0.01)
#py.plot(time[num1:num2],del_data_hall[num1:num2])
py.plot(time,data_hall_filt)
py.ylim([-0.01,0.01])
py.xlim([10,20])
py.xlabel('Time [sec]')
py.ylabel('Signal [a.u.]')
py.savefig(dir+'/output/'+runname+'hall_tod.png')
py.clf()


py.figure(2)
py.plot(freq_,psd_hall)
py.plot(freq_,psd_hall_filt)
py.plot(freq_,filt)
py.ylim([1e-6,10])
py.loglog()
py.xlabel('Frequency [Hz]')
py.ylabel('Signal [a.u.]')
py.savefig(dir+'/output/'+runname+'hall_psd.png')
py.clf()

sys.exit()

print num
print data1
print data1[num1:num2]
#py.plot(time[num1:num2], data1[num1:num2])

py.subplot(221)
py.plot(time, data1)
py.title('')
py.xlabel('Time [sec]')

py.subplot(223)
py.plot(psd1[0], psd1[1])
py.loglog()
py.xlabel('Frequency [Hz]')

py.subplot(222)
py.plot(time, data2)
py.xlabel('Time [sec]')
py.title('')

py.subplot(224)
py.plot(psd2[0], psd2[1])
py.loglog()
py.xlabel('Frequency [Hz]')

py.savefig(dir+'/output/tmp.png')

