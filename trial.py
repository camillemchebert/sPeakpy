# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:16:29 2022

@author: Geophys
"""



import specPeak as sp

import numpy as np
from tkinter.filedialog import askopenfilenames
import pandas as pd
from tkinter import Tk

def import_data():
    root = Tk(); root.withdraw()
    #"""Open a file for editing."""
    filepath = askopenfilenames(parent=root,
        filetypes=[("Excel Files", "*.csv"), ("Text Files", "*.txt"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")])
    if not filepath:
        return
        
    list_filepath=list(filepath)
    database=[]
   
    for file in list_filepath:

        f=open(file)
        f.read()
        f.close()

        data_p_values = pd.read_csv(file,delimiter=',', names=['Channel', 'Intensity'], skiprows = 21)
    
        de = 20.05*(10**-3)
    
        channel_p = data_p_values.Channel*de
        energy_p = (channel_p[channel_p<=24.0])
        intensity_p = (data_p_values.Intensity[channel_p<=24.0])
        database.append([energy_p, intensity_p])
        
    return database

input_data = import_data()

for i in np.arange(np.shape(input_data)[0]):
    energy = np.array(input_data[i][0]) 
    intensity = np.array(input_data[i][1])
    element_peaks = sp.Peak(energy, intensity, threshold = 60)
    




