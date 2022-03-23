from multiprocessing.dummy import Value
import numpy as np    
import matplotlib.pyplot as plt 
import csv
import pandas as pd
import math
from sympy import symbols, diff 
from scipy.optimize import curve_fit
import math

# Calculating the wavelength of He-Ne
# using the Fabry-Perot Method

files = ["f_p.csv", "m.csv"]

for file in files:
    file_data = pd.read_csv(file)
    start = file_data["Start"]
    end = file_data["Distance"]
    m = file_data["Count"]
    d_m = end-start
    wavelength = (2*d_m) / m
    avg_w = wavelength.mean()
    std_w = wavelength.std()
    #print("\n" + file + "\n" + str(wavelength))
    print("\n" + file + " Average Wavelength\n" + str(avg_w) + "\n" + str(std_w))
    

# Index of Refraction of Air Plot

def line(x,a,b):
    return a*x+b

air = pd.read_csv("air.csv")

Voltages = {100:['indigo','blue'],200:['teal','c'],300:['m','r'],400:['olive','olive'],500:['fuchsia','pink']}
#i is in mA, need to convert to A
em_values=[]
em_values_error=[]
for x in air["Count"]:
    pressure = air["Pressure"]
    m = air["Count"]
    wavel = 6.121669818936681e-5
    d = 3
    n = (m*wavel)/(2*d)
    error_n = (15*wavel)/(2*d)
    plt.errorbar(pressure, n, yerr=error_n,fmt='o',ecolor='black',color= 'r',capsize=5, label= "Air")
    popt, pcov = curve_fit(line,pressure,n,sigma=error_n)
    #print(popt,pcov)
    #print(f"A_",x," V =", popt[0], "+/-", pcov[0,0]**0.5)
    #print("b_",x, " V =", popt[1], "+/-", pcov[1,1]**0.5)
    em_values+=[(2*x)/((C*popt[0])**2)]
    #em_values_error+=[(pcov[0,0]**0.5)/(popt[0]**2)]

    xfine = np.arange(15,41,1)
    plt.plot(xfine, line(xfine, popt[0], popt[1]), color="m", label="air")
    plt.title("Least Squares Linear Fit of Required I With Respect to 1/r")
    plt.xlabel("Pressure (cm Hg)")
    plt.ylabel("N")
    #plt.legend(bbox_to_anchor =(1, 0.75))
#    plt.savefig("Graph1.png")
plt.savefig("air.png")
plt.show()