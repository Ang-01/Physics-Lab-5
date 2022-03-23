import numpy as np    
import matplotlib.pyplot as plt 
import csv
import pandas as pd
import math
from sympy import symbols, diff 


# Calculating the wavelength of He-N
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
    

# Index of Refraction of Air