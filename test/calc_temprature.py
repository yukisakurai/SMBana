import matplotlib.pyplot as plt
from pylab import *
import matplotlib.dates as mdates
import getpass

ch_number = 4
resistance = 108.2

dir = '/Users/' + getpass.getuser() + '/cernbox/LiteBIRD/analysis/data/'
constant_file = dir + 'temprature_constant.txt'

param = []

f = open(constant_file,'r')
i=0
for line in f.readlines():
    itemList = line[:-1].split(' ')
    temp = [itemList[1],itemList[2]]
    param.append(temp)
    i+=1

temprature = float(param[ch_number-1][0])*float(resistance)**float(param[ch_number-1][1])

str_ch     = 'ch_number   : ' + str(ch_number)
str_resist = 'resistance  : ' + str(resistance)
str_temp   = 'tempraturer : ' + str(round(temprature,3))
print str_ch
print str_resist
print str_temp
