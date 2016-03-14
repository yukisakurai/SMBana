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



#x = linspace(0.505, 1.65, 10000)
print lib_temp.CalibSiPD([1.009200],'D6059358')

# print lib_temp.CalibSiPD(0.8,'D6059358')
# plt.plot(y, x, "o")
# plt.show()

