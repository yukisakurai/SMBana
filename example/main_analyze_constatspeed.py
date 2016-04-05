import numpy as np
import pylab as py
import lib_smb as lib_s
import lib_m as lib_m
from scipy.optimize import curve_fit
import sys
import os

'''
main_analyze_rot.py
Written by T. Matsumura, 2015-3-19

'''

# DEFINE THE FUNCTIONS
def model_exp_spin_down(x,a,b,c):
	model = a*np.exp(-(x-b)*c)
	return model

def model_explin_spin_down(x,a,b,c,d):
	model = a*np.exp(-(x-b)*c)+d
	return model

def model_lin(x,a,b):
	model = a*x+b
	return model


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# DEFINE THE BASIC PARAMETERS++++++++++++++++++++++++++++++++++++++++++++++
pi = np.pi
num_slots = 60.


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# READ THE DATA++++++++++++++++++++++++++++++++++++++++++++++
print ''
print 'READ THE DATA'

dir = '/Users/tomotake_matsumura/Documents/Projects/LiteBIRD/20150319_SMBBBM_LN2/'
dir_data = dir + '/data/20150319_TM_Processed/'
runname = 'run003_2_txttab'
data_hall, data_enco = lib_m.read_txt2f_skipn(dir_data+runname+'.txt',1)

num = len(data_hall)
d_time = 1.e-3
time = np.arange(num)*d_time
samplerate = 1./d_time


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ANALYZE THE OPTICAL ENCODER DATA++++++++++++++++++++++++++++++++++++++++++++++
print ''
print 'ANALYZE THE OPTICAL ENCODER DATA'

#filtertype = 'cosine_lowpass'
#par = [20.,20]
#filtertype = 'cosine_bandpass'
#par = [1.,0.2, 30., 5.]
#freq_, psd_enco, psd_enco_filt, data_enco_filt, filt = lib_m.AppFilt(data_enco-np.mean(data_enco), samplerate, 6, filtertype, par)
#filtertype = 'cosine_highpass'
#par = [1.,0.2]
#freq_, psd_enco, psd_enco_filt, data_enco_filt, filt = lib_m.AppFilt(data_enco_filt, samplerate, 6, filtertype, par)
del_data_enco = data_enco - np.mean(data_enco)
flag = np.zeros(num,dtype='int')
j = 0
for i in range(1,num):
	if ((del_data_enco[i]>0) & (del_data_enco[i-1]<0)): flag[i] = 1
	if ((del_data_enco[i]<0) & (del_data_enco[i-1]>0)): flag[i] = -1

ind = np.where(flag == 1)
num_ind = len(ind[0])
rot_freq = np.zeros((2,num_ind-1))
angle = np.zeros((2,num_ind-1))
angle_i = 0.; angle_wrap_i = 0.
for i in range(num_ind-1):
	rot_freq[0,i] = 0.5*(time[ind[0][i+1]]+time[ind[0][i]])
	rot_freq[1,i] = 1./(time[ind[0][i+1]]-time[ind[0][i]])/float(num_slots)
	print time[ind[0][i]], time[ind[0][i+1]], time[ind[0][i+1]]-time[ind[0][i]], 1./(time[ind[0][i+1]]-time[ind[0][i]]), 1./(time[ind[0][i+1]]-time[ind[0][i]])/float(num_slots)
	angle_i += 6. 
	angle_wrap_i += 6.
	angle[0,i] += angle_i
	angle[1,i] += angle_wrap_i
	if (angle_i%360) == 0:
		angle_wrap_i = 0.
num_median = 1
rot_freq_mfilt = lib_m.median_filter(rot_freq[1],num_median)

flag_rev = np.zeros(num_ind-1,dtype='int')
for i in range(0,num_ind-2):
	if ((angle[1,i+1]-angle[1,i]) < 0):
		flag_rev[i+1] = 1

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FIT TO THE SPIN DOWN RATE++++++++++++++++++++++++++++++++++++++++++++++
x_ = rot_freq[0][num_median:num-num_median]
y_ = rot_freq_mfilt[0:num]
par_guess_exp = np.array([0.55,0.,1./400.])
popt_exp, pcov_exp = curve_fit(model_exp_spin_down, x_, y_, p0=par_guess_exp)
print 'Fit results from exp only'
print popt_exp
print pcov_exp

par_guess_explin = np.array([0.55,0.,1./400.,0.])
popt_explin, pcov_explin = curve_fit(model_explin_spin_down, x_, y_, p0=par_guess_explin)
print 'Fit results from exp + const'
print popt_explin
print pcov_explin
a1 = popt_explin[2]
a0 = -2.*pi*popt_explin[3]*popt_explin[2]
f0 = popt_explin[0] + popt_explin[3]
print 'a1:', a1
print 'a0:', a0
print 'f0:', f0

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# OUTPUT THE PLOTS++++++++++++++++++++++++++++++++++++++++++++++
num1 = 10000
num2 = 10100

py.figure(1)
py.subplot(211)
py.plot(time,del_data_enco)
py.plot(time,flag*0.01)
py.xlabel('Time [sec]')
py.subplot(212)
py.plot(time[num1:num2],del_data_enco[num1:num2],'-')
py.plot(time[num1:num2],del_data_enco[num1:num2],'.')
py.plot(time[num1:num2],flag[num1:num2]*0.01)
py.xlabel('Time [sec]')
py.ylabel('Signal [a.u.]')
py.savefig(dir+'/output/'+runname+'optenc_tod.png')
py.clf()

#py.figure(2)
#py.plot(freq_,psd_enco)
#py.plot(freq_,psd_enco_filt)
#py.plot(freq_,filt)
#py.ylim([1e-6,10])
#py.loglog()
#py.xlabel('Frequency [Hz]')
#py.ylabel('Signal [a.u.]')
#py.savefig(dir+'/output/'+runname+'optenc_psd.png')
#py.clf()

py.figure(3)

py.subplot(221)
#py.plot(rot_freq[0][num_median:num-num_median-1000],rot_freq_mfilt[0:num-1000],'.')
py.plot(rot_freq[0],rot_freq[1],'.')

num_fit = 100
freq_fit = np.arange(num_fit)/float(num_fit)*max(rot_freq[0])
#fit_exp = model_exp_spin_down(freq_fit,popt_exp[0],popt_exp[1],popt_exp[2])
fit_explin = model_explin_spin_down(freq_fit,popt_explin[0],popt_explin[1],popt_explin[2],popt_explin[3])
#py.plot(freq_fit,fit_exp)

py.plot(freq_fit,fit_explin,'r')
py.ylabel('Rot. rate [Hz]')
#py.xlabel('Time [sec]')
py.ylim([0, f0*1.2 ])


py.subplot(222)
popt_lin, pcov_lin = curve_fit(model_lin, rot_freq[0],angle[0])
print popt_lin
print pcov_lin
py.plot(rot_freq[0],angle[0],'.')
x_ = np.arange(num_fit)/float(num_fit)*max(rot_freq[0])
py.plot(x_, model_lin(x_,popt_lin[0],popt_lin[1]),'r-')
py.ylabel('Rot. angle [degs]')
py.xlabel('Time [sec]')

ind_ = np.where(flag_rev == 1)
num_ind = len(ind_[0])
for i in range(0,num_ind-1):
	i_i = ind_[0][i]
	i_f = ind_[0][i+1]
	popt_lin, pcov_lin = curve_fit(model_lin, rot_freq[0][i_i:i_f], angle[1][i_i:i_f])

	py.subplot(223)
	py.plot(rot_freq[0][i_i:i_f], angle[1][i_i:i_f],'.')
	model_ = model_lin(rot_freq[0][i_i:i_f], popt_lin[0], popt_lin[1])
	py.plot(rot_freq[0][i_i:i_f], model_,'-')
	py.xlabel('Time [sec]')
	py.ylabel('Rot. angle [degs]')

	py.subplot(224)
	py.plot(rot_freq[0][i_i:i_f], angle[1][i_i:i_f]-model_,'-')
	residual_tmp = angle[1][i_i:i_f]-model_
	if i==0: residual_ = angle[1][i_i:i_f]-model_
	residual_ = np.hstack((residual_,residual_tmp))
	py.xlabel('Time [sec]')
#	py.ylabel('Rot. angle res. [degs]')

py.savefig(dir+'/output/'+runname+'optenc_travel.png')
py.clf()

py.figure(10)
py.subplot(121)
parout_hist = lib_m.plot_hist(residual_,30,fit=True,init_auto=True,xtitle=-1,no_plot=False)
py.title('$(\mu, \sigma)$=(%1.3f, %1.3f) degs' % (parout_hist[1], parout_hist[2]))
py.xlabel('Reconstructed angle [degs]')
py.savefig(dir+'/output/'+runname+'optenc_reshist.png')
py.clf()

#py.subplot(224)
#py.plot(rot_freq[0], angle[0] - model_lin(rot_freq[0],popt_lin[0],popt_lin[1]) ,'.')
#py.ylabel('del. Rot. angle [degs]')


py.figure(4)
py.subplot(211)
py.plot(rot_freq[0],angle[1],'-')
py.xlabel('Time [sec]')
py.ylabel('Wrap. angle [degs]')
py.subplot(211)
py.plot(rot_freq[0][num1:num2],angle[1][num1:num2],'-')
py.ylabel('Wrap. angle [degs]')
py.xlabel('Time [sec]')
py.savefig(dir+'/output/'+runname+'optenc_wraptravel.png')
py.clf()




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

