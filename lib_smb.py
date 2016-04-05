import os
import math
import datetime
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
import inspect
from tqdm import tqdm

def gaussian(x, norm, mu, sigma):
    return norm * exp(-(x - mu)**2.0 / (2 * sigma**2))

def linear(x,a,b):
    return a*x + b

def sawtooth(x,a,b):

    flag = np.zeros(len(x))
    int = 0
    for i in range(len(x)):
        if x[i] - int > 1.0:
            flag[i-1] = 1
            int += 1

    y_new = np.zeros(len(x))
    int = 0
    for i in range(len(x)):
        if flag[i] == 1:
            y_new[i] = 0.0
        else: y_new[i] = a * (x[i] - math.floor(x[i])) + b
    return y_new

def exponential(x,a,b,c):
    return a * np.exp( -b * x ) + c

def spindown(x,a,b):
    return a + b / (x * x)

def pol2(x,a,b,c):
    return a + b*x + c*x*x

def read_data_1ch(filename,start,sampling):

    time = np.array([])
    arr1 = np.array([]).astype(np.float)

    f = open(filename,'r')
    data = f.read()
    lines = data.split('\n')
    lines.remove('')
    print len(lines)
    i = 0
    for line in lines:
        i += 1
        if i <= start: continue

        # define data
        line = line.rstrip()
        item = line.split('\t')
        DateTime = item[0].split(' ')
        Date = DateTime[0].split('/')
        Time = DateTime[1].split(':')
        year = 2000 + int(Date[0])
        month = int(Date[1])
        day = int(Date[2])
        hour = int(Time[0])
        min = int(Time[1])
        sec = int(Time[2].split('.')[0])
        msec = int(Time[2].split('.')[1])
        DateTime = datetime.datetime(year, month, day, hour=hour, minute=min, second=sec, microsecond=msec)

        # add data to arrays
        time = np.append(time,DateTime)
        arr1 = np.append(arr1,float(item[1]))

        if i >= sampling + start: break

    return time, arr1

def read_data_2ch(filename,sampling):
    max_sample = 1000000000

    time = np.array([])
    arr1 = np.array([]).astype(np.float)
    arr2 = np.array([]).astype(np.float)

    f = open(filename,'r')
    data = f.read()
    lines = data.split('\r')
    lines.remove('\n')

    i = 0
    for line in lines:
        i += 1

        # define data
        line = line.rstrip()
        item = line.split('\t')
        DateTime = item[0].split(' ')
        Date = DateTime[0].split('/')
        Time = DateTime[1].split(':')
        year = 2000 + int(Date[0])
        month = int(Date[1])
        day = int(Date[2])
        hour = int(Time[0])
        min = int(Time[1])
        sec = int(Time[2].split('.')[0])
        msec = int(Time[2].split('.')[1])
        DateTime = datetime.datetime(year, month, day, hour=hour, minute=min, second=sec, microsecond=msec)

        # add data to arrays
        time = np.append(time,DateTime)
        arr1 = np.append(arr1,float(item[1]))
        arr2 = np.append(arr2,float(item[2]))

        if sampling != -1:
            if i >= sampling: break

    return time, arr1, arr2


def read_data_3ch(filename,sampling):
    max_sample = 10000000

    time = np.array([])
    arr1 = np.array([])
    arr2 = np.array([])
    arr3 = np.array([])

    f = open(filename,'r')
    data = f.read()
    lines = data.split('\n')
    lines.remove('')

    i = 0
    for line in lines:
        i += 1


        # define data
        line = line.rstrip()
        item = line.split('\t')
        DateTime = item[0].split(' ')

        Date = DateTime[0].split('/')
        Time = DateTime[1].split(':')

        year = int(Date[0])
        month = int(Date[1])
        day = int(Date[2])
        hour = int(Time[0])
        min = int(Time[1])
        sec = int(Time[2].split('.')[0])
        msec = int(Time[2].split('.')[1])

        DateTime = datetime.datetime(year, month, day, hour=hour, minute=min, second=sec, microsecond=msec)

        # add data to arrays
        time = np.append(time,DateTime)
        arr1 = np.append(arr1,float(item[1]))
        arr2 = np.append(arr2,float(item[2]))
        arr3 = np.append(arr3,float(item[3]))


        if sampling != -1:
            if i >= sampling+1000: break
        else:
            if i >= max_sample: break

    return time, arr1, arr2, arr3


def read_TM(filename,sampling):

    time = np.array([])
    ch1 = np.array([])
    ch2 = np.array([])
    ch3 = np.array([])
    ch4 = np.array([])
    ch5 = np.array([])
    ch6 = np.array([])
    ch7 = np.array([])
    ch8 = np.array([])

    f = open(filename,'r')
    data = f.read()
    lines = data.split('\n')
    lines.remove('')

    i = 0
    for line in lines:
        if i > sampling: break
        i += 1

        # define data
        line = line.rstrip()
        item = line.split('\t')
        Date = item[0].split('/')
        Time = item[1].split(':')

        year = int(Date[0])
        month = int(Date[1])
        day = int(Date[2])
        hour = int(Time[0])
        min = int(Time[1])
        sec = int(Time[2].split('.')[0])

        DateTime = datetime.datetime(year, month, day, hour=hour, minute=min, second=sec)
        # add data to arrays
        time = np.append(time,DateTime)
        ch1 = np.append(ch1,float(item[2]))
        ch2 = np.append(ch2,float(item[3]))
        ch3 = np.append(ch3,float(item[4]))
        ch4 = np.append(ch4,float(item[5]))
        ch5 = np.append(ch5,float(item[6]))
        ch6 = np.append(ch6,float(item[7]))
        ch7 = np.append(ch7,float(item[8]))
        ch8 = np.append(ch8,float(item[9]))

    ch = []
    ch.append(ch1)
    ch.append(ch2)
    ch.append(ch3)
    ch.append(ch4)
    ch.append(ch5)
    ch.append(ch6)
    ch.append(ch7)
    ch.append(ch8)
    ch = np.array(ch)

    return time, ch

def read_TBM(filename, sampling):

    time = np.array([])
    ch1 = np.array([])
    ch2 = np.array([])
    ch3 = np.array([])
    ch4 = np.array([])
    ch5 = np.array([])
    ch6 = np.array([])
    ch7 = np.array([])
    ch8 = np.array([])
    B = np.array([])

    f = open(filename,'r')
    data = f.read()
    lines = data.split('\n')
    lines.remove('')

    i = 0
    for line in lines:
        if i > sampling: break
        i += 1

        # define data
        line = line.rstrip()
        item = line.split('\t')
        Date = item[0].split('/')
        Time = item[1].split(':')

        year = int(Date[0])
        month = int(Date[1])
        day = int(Date[2])
        hour = int(Time[0])
        min = int(Time[1])
        sec = int(Time[2].split('.')[0])

        DateTime = datetime.datetime(year, month, day, hour=hour, minute=min, second=sec)
        # add data to arrays
        time = np.append(time,DateTime)
        ch1 = np.append(ch1,float(item[2]))
        ch2 = np.append(ch2,float(item[3]))
        ch3 = np.append(ch3,float(item[4]))
        ch4 = np.append(ch4,float(item[5]))
        ch5 = np.append(ch5,float(item[6]))
        ch6 = np.append(ch6,float(item[7]))
        ch7 = np.append(ch7,float(item[8]))
        ch8 = np.append(ch8,float(item[9]))
        B = np.append(B,float(item[10]))

    ch = []
    ch.append(ch1)
    ch.append(ch2)
    ch.append(ch3)
    ch.append(ch4)
    ch.append(ch5)
    ch.append(ch6)
    ch.append(ch7)
    ch.append(ch8)
    ch = np.array(ch)

    return time, ch, B

def read_TM_6ch(filename,sampling):

    time = np.array([])
    ch1 = np.array([])
    ch2 = np.array([])
    ch3 = np.array([])
    ch4 = np.array([])
    ch5 = np.array([])
    ch6 = np.array([])

    f = open(filename,'r')
    data = f.read()
    lines = data.split('\n')
    lines.remove('')

    i = 0
    for line in lines:
        if i > sampling: break
        i += 1


        # define data
        line = line.rstrip()
        item = line.split('\t')
        Date = item[0].split('/')
        Time = item[1].split(':')

        year = int(Date[0])
        month = int(Date[1])
        day = int(Date[2])
        hour = int(Time[0])
        min = int(Time[1])
        sec = int(Time[2].split('.')[0])

        DateTime = datetime.datetime(year, month, day, hour=hour, minute=min, second=sec)

        # add data to arrays
        time = np.append(time,DateTime)
        ch1 = np.append(ch1,float(item[2]))
        ch2 = np.append(ch2,float(item[3]))
        ch3 = np.append(ch3,float(item[4]))
        ch4 = np.append(ch4,float(item[5]))
        ch5 = np.append(ch5,float(item[6]))
        ch6 = np.append(ch6,float(item[7]))

    ch = []
    ch.append(ch1)
    ch.append(ch2)
    ch.append(ch3)
    ch.append(ch4)
    ch.append(ch5)
    ch.append(ch6)
    ch = np.array(ch)

    return time, ch

def read_VTBM(filename, sampling):

    time = np.array([])
    ch1 = np.array([])
    ch2 = np.array([])
    ch3 = np.array([])
    ch4 = np.array([])
    ch5 = np.array([])
    ch6 = np.array([])
    ch7 = np.array([])
    ch8 = np.array([])
    ch5_T = np.array([])
    ch6_T = np.array([])
    ch7_T = np.array([])
    ch8_T = np.array([])
    B = np.array([])

    f = open(filename,'r')
    data = f.read()
    lines = data.split('\n')
    lines.remove('')

    i = 0
    for line in lines:
        if i > sampling: break
        i += 1

        # define data
        line = line.rstrip()
        item = line.split('\t')
        Date = item[0].split('/')
        Time = item[1].split(':')

        year = int(Date[0])
        month = int(Date[1])
        day = int(Date[2])
        hour = int(Time[0])
        min = int(Time[1])
        sec = int(Time[2].split('.')[0])

        DateTime = datetime.datetime(year, month, day, hour=hour, minute=min, second=sec)
        # add data to arrays
        time = np.append(time,DateTime)
        ch1 = np.append(ch1,float(item[2]))
        ch2 = np.append(ch2,float(item[3]))
        ch3 = np.append(ch3,float(item[4]))
        ch4 = np.append(ch4,float(item[5]))
        ch5 = np.append(ch5,float(item[6]))
        ch6 = np.append(ch6,float(item[7]))
        ch7 = np.append(ch7,float(item[8]))
        ch8 = np.append(ch8,float(item[9]))
        ch5_T = np.append(ch5_T,float(item[10]))
        ch6_T = np.append(ch6_T,float(item[11]))
        ch7_T = np.append(ch7_T,float(item[12]))
        ch8_T = np.append(ch8_T,float(item[13]))
        B = np.append(B,float(item[14]))

    ch = []
    ch.append(ch1)
    ch.append(ch2)
    ch.append(ch3)
    ch.append(ch4)
    ch.append(ch5)
    ch.append(ch6)
    ch.append(ch7)
    ch.append(ch8)
    ch = np.array(ch)

    ch58_T = []
    ch58_T.append(ch5_T)
    ch58_T.append(ch6_T)
    ch58_T.append(ch7_T)
    ch58_T.append(ch8_T)
    ch58_T = np.array(ch58_T)

    return time, ch, ch58_T, B

def read_VTRBM(filename, sampling):

    time = np.array([])
    ch1 = np.array([])
    ch2 = np.array([])
    ch3 = np.array([])
    ch4 = np.array([])
    ch5 = np.array([])
    ch6 = np.array([])
    ch7 = np.array([])
    ch5_T = np.array([])
    ch6_T = np.array([])
    ch7_T = np.array([])
    ch8_R = np.array([])
    B = np.array([])

    f = open(filename,'r')
    data = f.read()
    lines = data.split('\n')
    lines.remove('')

    i = 0
    for line in tqdm(lines):
        if i > sampling: break
        i += 1

        # define data
        line = line.rstrip()
        item = line.split('\t')
        Date = item[0].split('/')
        Time = item[1].split(':')

        year = int(Date[0])
        month = int(Date[1])
        day = int(Date[2])
        hour = int(Time[0])
        min = int(Time[1])
        sec = int(Time[2].split('.')[0])

        DateTime = datetime.datetime(year, month, day, hour=hour, minute=min, second=sec)
        # add data to arrays
        time = np.append(time,DateTime)
        ch1 = np.append(ch1,float(item[2]))
        ch2 = np.append(ch2,float(item[3]))
        ch3 = np.append(ch3,float(item[4]))
        ch4 = np.append(ch4,float(item[5]))
        ch5 = np.append(ch5,float(item[6]))
        ch6 = np.append(ch6,float(item[7]))
        ch7 = np.append(ch7,float(item[8]))
        ch5_T = np.append(ch5_T,float(item[9]))
        ch6_T = np.append(ch6_T,float(item[10]))
        ch7_T = np.append(ch7_T,float(item[11]))
        ch8_R = np.append(ch8_R,float(item[12]))
        B = np.append(B,float(item[13]))

    ch = []
    ch.append(ch1)
    ch.append(ch2)
    ch.append(ch3)
    ch.append(ch4)
    ch.append(ch5)
    ch.append(ch6)
    ch.append(ch7)
    ch = np.array(ch)

    ch57_T = []
    ch57_T.append(ch5_T)
    ch57_T.append(ch6_T)
    ch57_T.append(ch7_T)
    ch57_T = np.array(ch57_T)

    return time, ch, ch57_T, ch8_R, B


def get_calib_TM(R):

    T = np.array([])
    Calib = np.load('20160205_Calib.npz')
    T_calib = Calib['T']
    R_calib = Calib['R']

    R_sort = np.sort(R_calib)
    R_sort_idx = np.argsort(R_calib)
    T_sort = T_calib[R_sort_idx]

    for i in range(len(R)):
        bit = 0
        for j in range(len(R_sort)):
            if j == 0:
                if R[i] <= R_sort[0]:
                    T = np.append(T,T_sort[0])
                    bit += 1
                    continue
            else :
                if R[i] <= R_sort[j] and R[i] > R_sort[j-1]:
                    T = np.append(T,T_sort[j])
                    bit += 1
                    continue
        if R[i] > R_sort[len(R_sort)-1]:
            T = np.append(T,T_sort[len(T_sort)-1])
            bit += 1
            continue


    return T

def get_rel_time(datetime):

    rel_time = np.zeros(len(datetime)).astype(float)
    for i in range(0,len(datetime)):
        rel_time[i] = round(  (datetime[i].day - datetime[0].day) * 24 * 60 * 60 + \
                                  (datetime[i].hour - datetime[0].hour) * 60 * 60 + \
                                  (datetime[i].minute - datetime[0].minute) * 60 + \
                                  (datetime[i].second - datetime[0].second) + \
                                  (datetime[i].microsecond - datetime[0].microsecond) * 1e-6 , 10 )
    return rel_time

def get_rel_time_min(datetime):

    rel_time = np.zeros(len(datetime)).astype(float)
    for i in range(0,len(datetime)):
        rel_time[i] = round(  (datetime[i].day - datetime[0].day) * 24 * 60+ \
                                  (datetime[i].hour - datetime[0].hour) * 60 + \
                                  (datetime[i].minute - datetime[0].minute) + \
                                  (datetime[i].second - datetime[0].second) / 60. + \
                                  (datetime[i].microsecond - datetime[0].microsecond) * 1e-6 / 60., 10 )
    return rel_time

def get_rel_time_hour(datetime):

    rel_time = np.zeros(len(datetime)).astype(float)
    for i in range(0,len(datetime)):
        rel_time[i] = round(  (datetime[i].day - datetime[0].day) * 24 + \
                                  (datetime[i].hour - datetime[0].hour) + \
                                  (datetime[i].minute - datetime[0].minute) / 60. + \
                                  (datetime[i].second - datetime[0].second) / 60. / 60. + \
                                  (datetime[i].microsecond - datetime[0].microsecond) * 1e-6 / 60. / 60., 10 )
    return rel_time

def get_del_time(datetime1,datetime2):

    del_time = round( (datetime2.hour - datetime1.hour) * 60 * 60 + \
                              (datetime2.minute - datetime1.minute) * 60 + \
                              (datetime2.second - datetime1.second) + \
                              (datetime2.microsecond - datetime1.microsecond) * 1e-6 , 10 )
    return del_time

def get_ave_time(datetime):

    sum_time = 0
    for i in range(1,len(time)):
        sum_time += del_time(datetime[i+1],datetime[i])
    ave_time = datetime[0] + sum_time / float(len(datetime))

    return ave_time

def get_opt_start_point(time, opt):

    reltime = get_rel_time(time)
    delta_opt = opt - opt.mean()
    thr = 0

    num = len(opt)
    start_point = np.array([])
    start_time = np.array([])
    for i in range(1,num-4):
	if ((delta_opt[i]>thr) & \
                (delta_opt[i+1]<thr) & \
                (delta_opt[i+2]<thr) & \
                (delta_opt[i+3]<thr) & \
                (delta_opt[i+4]<thr)):
            # linear fit with 2 points ===
            x1 = [ reltime[i], reltime[i+1] ]
            y1 = [ delta_opt[i], delta_opt[i+1] ]
            popt, pcov = optimize.curve_fit(linear, x1, y1)
            fit_1 = -popt[1] / popt[0]
            start_time = np.append(start_time,fit_1)
            start_point = np.append(start_point, popt[0] * -popt[1] / popt[0] + popt[1])

    return start_time, start_point

def get_opt_freq(num_slots, time, opt):

    reltime = get_rel_time(time)
    delta_opt = opt - opt.mean()
    thr = 0

    num = len(opt)
    start_point_time = np.array([])
    for i in range(1,num-4):
        if ((delta_opt[i]>thr) & \
                (delta_opt[i+1]<thr) & \
                (delta_opt[i+2]<thr) & \
                (delta_opt[i+3]<thr) & \
                (delta_opt[i+4]<thr)):
            # linear fit with 2 points ===
            a = (delta_opt[i+1] - delta_opt[i]) / (reltime[i+1] - reltime[i])
            b =  delta_opt[i] - a * reltime[i]
            start_point_time = np.append(start_point_time, -b / a)

    num = len(start_point_time)
    freq_time = np.array([])
    freq = np.array([])
    angle = np.array([])
    theta = 0
    for i in range(num-1):

        dtime = start_point_time[i+1] - start_point_time[i]
        if len(freq) > 0:
            if 1./(dtime * num_slots) > freq[len(freq)-1] * 2: continue

        freq_time = np.append(freq_time,start_point_time[i])
        freq = np.append(freq,1./(dtime * num_slots))

        theta += 360 * 1./(dtime * num_slots) * dtime
        if theta%360==0: theta = 0
        angle = np.append(angle,theta)

    return freq_time, freq, angle

def get_opt_freq_period(num_slots, time, opt):

    reltime = get_rel_time(time)
    delta_opt = opt - opt.mean()
    thr = 0

    num = len(opt)
    start_point_time = np.array([])
    for i in range(1,num-4):
        if ((delta_opt[i]>thr) & \
                (delta_opt[i+1]<thr) & \
                (delta_opt[i+2]<thr) & \
                (delta_opt[i+3]<thr) & \
                (delta_opt[i+4]<thr)):
            # linear fit with 2 points ===
            a = (delta_opt[i+1] - delta_opt[i]) / (reltime[i+1] - reltime[i])
            b =  delta_opt[i] - a * reltime[i]
            start_point_time = np.append(start_point_time, -b / a)

            num = len(start_point_time)
            freq_time = np.array([])

    freq = np.array([])
    freq_period = np.array([])
    freq_time_period = np.array([])
    delta_freq = np.array([])
    angle = np.array([])
    theta = 0
    for i in range(num-1):

        dtime = start_point_time[i+1] - start_point_time[i]
        frequency = 1./(dtime * num_slots)
        if len(freq) > 0:
            if frequency > freq[len(freq)-1] * 2: continue

        freq = np.append(freq,frequency)
        theta += 360. / float(num_slots)
        if theta%360==0:
            theta = 0
            freq_time_period = np.append(freq_time_period,start_point_time[i])
            freq_period = np.append(freq_period,freq.mean())
            delta_freq = np.append(delta_freq,(freq.max()-freq.min())/frequency)
            freq = np.array([])

    return freq_time_period, freq_period, delta_freq

def get_driver_RPM(time, driver):

    reltime = get_rel_time(time)

    delta_driver = driver - driver.mean()

    pulse = np.array([])
    pulse_time = np.array([])
    for i in range(0,len(delta_driver)):

        if (delta_driver[i-1] < 0)      and \
                (delta_driver[i] > 0)   and \
                (delta_driver[i+1] > 0) :
            pulse = np.append(pluse, delta_driver[i])
            pulse_time = np.append(pulse_time, reltime[i])

    RPM = np.array([])
    RPM_time = np.array([])
    for i in range(0,len(pulse)-1):
        interval = pulse_time[i+1] - pulse_time[i]
        rpm = 1. / (interval * 30) * 60
        RPM = np.append(RPM,rpm)
        RPM_time = np.append(RPM_time, pulse_time[i])

    return RPM_time, RPM

def get_driver_freq(gear_ratio, time, driver):

    reltime = get_rel_time(time)
    delta_driver = driver - driver.mean()

    pulse = np.array([])
    pulse_time = np.array([])
    for i in range(0,len(delta_driver)):

        if (delta_driver[i-1] < 0)      and \
                (delta_driver[i] > 0)   and \
                (delta_driver[i+1] > 0) :
            pulse = np.append(pulse, delta_driver[i])
            pulse_time = np.append(pulse_time, reltime[i])

    freq = np.array([])
    freq_time = np.array([])
    angle = np.array([])
    theta = 0.0
    for i in range(0,len(pulse)-1):
        dtime = pulse_time[i+1] - pulse_time[i]
        RPM = 1. / (dtime * 30) * 60
        freq = np.append(freq,RPM/60. * gear_ratio)
        freq_time = np.append(freq_time,pulse_time[i])
        theta += 360 / 30. * gear_ratio
        if theta%360==0: theta = 0.0
        angle = np.append(angle,theta)

    return freq_time, freq, angle


def get_linear_fit_array(object_array,object_time,target_array,target_time):

    num_elements = len(object_array)
    num_target_elements = len(target_array)
    period = num_elements / num_target_elements

    array = np.array([])
    elements = np.array([])
    elements_time = np.array([])
    idx = 0
    offset = 0.0
    for i in range(num_elements):
        if len(array) == len(target_array): break
        elements = np.append(elements,object_array[i])
        elements_time = np.append(elements_time,object_time[i])
        if (i%period == 0) & (i!=0):

            # Adjust list_elements =====
            return_idx = -1
            for j in range(len(elements)):
                if elements[j] == 0.0: return_idx = j
            if return_idx != -1:
                for j in range(return_idx):
                    elements[j] = elements[j] - 360.0

            # Linear fit =====
            popt, pcov = optimize.curve_fit(linear, elements_time, elements)
            fit_result = popt[0]*target_time[idx] + popt[1]

            # Define offset =====
            if idx == 0:
                offset = target_array[0] - fit_result
            fit_result += offset

            array = np.append(array,fit_result)

            elements = np.array([])
            elements_time = np.array([])
            idx += 1

    return array

def get_average_array(object_array,target_array):

    num_elements = len(object_array)
    num_target_elements = len(target_array)
    period = num_elements / num_target_elements

    array = np.array([])
    elements = np.array([])
    idx = 0
    for i in range(1,num_elements):
        if len(array) == len(target_array): break
        elements = np.append(elements,object_array[i])
        if i%period == 0:
            array_elements = np.array(elements)
            array = np.append(array,array_elements.mean())
            elements = np.array([])
            array_elements = np.array([])
            idx += 1

    return array

def get_adjust_array(object_array,object_time,target_array,target_time):

    arr = np.array([])
    array = np.array([])
    for i in range(0,len(target_time)):
        for j in range(0,len(object_time)):
            if i==0:
                if j==0: arr = np.append(array,object_array[j])
                else : continue
            else :
                if (object_time[j]<target_time[i]) and (object_time[j]>target_time[i-1]):
                    arr = np.append(array,object_array[j])
        array = np.append(array,arr[len(arr)-1])

    return array
