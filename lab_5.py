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

files = {"f_p.csv":"Fabry-Perot", "m.csv": "Michaelson"}

for file in files:
    file_data = pd.read_csv(file)
    start = file_data["Start"]
    end = file_data["Distance"]
    m = file_data["Count"]
    d_m = end-start
    wavelength = (2*d_m) / m
    avg_w = 1000 * wavelength.mean()
    std_w = 1000 * wavelength.std()
    #print("\n" + file + "\n" + str(wavelength))
    print(files[file] + " Average Wavelength = " + str(avg_w) + " +/- " + str(std_w) + " nm")
 
print("Accepted Value = 632.9 nm")

# Index of Refraction of Air Plot

def line(x,a,b):
    return a*x+b
print('\n')
air = pd.read_csv("air.csv")

pressure = air["Pressure"]
m = air["Count"]
wavel = 6.328e-5
d = 3
n = (m*wavel)/(2*d)
error_n = (5*wavel)/(2*d)
plt.errorbar(pressure, n, yerr=error_n,fmt='o',ecolor='black',color= 'm',capsize=5, label= "Air")
popt, pcov = curve_fit(line,pressure,n,sigma=[error_n]*7)
#print(f"C =", popt[0], "+/-", pcov[0,0]**0.5)
C=popt[0]
C_err = pcov[0,0]**0.5
#print("b_",x, " V =", popt[1], "+/-", pcov[1,1]**0.5)
#em_values_error+=[(pcov[0,0]**0.5)/(popt[0]**2)]

xfine = np.arange(0,25,1)
plt.plot(xfine, line(xfine, popt[0], popt[1]), color="b", label="air")
plt.title("Least Squares Linear Fit of n with respect to P")
plt.xlabel("Guage Pressure (cm Hg)")
plt.ylabel("Nf-Ni")
plt.grid()
#plt.legend(bbox_to_anchor =(1, 0.75))
#plt.savefig("Graph1.png")
plt.savefig("air.png")
plt.show()

print(f"C =", C, "+/-", C_err)

n_atm = 1+C*76
n_atm_err = C_err*76

print(f"n_atm =",n_atm,"+/-",n_atm_err)
print("Accepted Value For Air = 1.00029")

glass = pd.read_csv("glass.csv")

n_air = 1.00028
d_g=glass['d (mm)']/1000
gamma = wavel/100
m_g=glass['Count']
theta = glass['angle']*2*math.pi/360

n_glist = n_air*(2*d_g*n_air - m_g*gamma)*(1-np.cos(theta))/(2*d_g*n_air*(1-np.cos(theta))-m_g*gamma)

n_glass = n_glist.mean()
n_glass_err = n_glist.std()

print(f"n_glass =",n_glass,"+/-",n_glass_err)