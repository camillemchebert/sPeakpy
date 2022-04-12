import sys
import os
import pandas as pd
import numpy as np
from tkinter.filedialog import askopenfilenames
from specPeak.Preprocessing import Preprocessing
from specPeak.Segmentation import Segmentation
from specPeak.Classification import Classification
from specPeak.Element_ID import Element_ID
import matplotlib.pyplot as plt
from tkinter import Tk



class Peak:
    def __init__(self, 
                 energy=[], intensity=[], filter_type='mov_avg', 
                 ws=11, threshold=70, element_ID = True, DataType='EDXRF',
                 output_data = None, plot_data=True):
        
        if len(energy)==0 or len(intensity)==0:
            input_data = self.import_data()
            energy = input_data[0][0]
            intensity = input_data[0][1]
        
        self.r=Preprocessing(energy, intensity, filter_type='moving_avg', ws=9)
        self.s=Segmentation(self.r.signal_)
        self.c=Classification(self.s.index_segment_, threshold)
        self.e=Element_ID(self.c.index_segment_bin, DataType)

        if plot_data==True:
            plt.plot(energy, intensity, color='lightgray', alpha=0.7)
            for i in self.c.index_segment_bin:
                plt.plot(energy[i], intensity[i])
            plt.xlabel('Energy (keV)')
            plt.ylabel('Intensity (counts)')
            plt.show()
            

        if not output_data == None:
            self.get_data(str(output_data))

	
    def get_data(self, fileName=None):
        if fileName == None:
            print(self.e.EDXRF_ID())
     
        else:
            if not os.path.isdir('Results'):
                os.mkdir('Results')
            if not os.path.isdir(os.path.join('Results', fileName+'.csv')):
                os.mkdir(os.path.join('Results',fileName+'.csv'))
                
    def import_data(self):
        root = Tk(); root.withdraw()
        #"""Open a file for editing."""
        filepath = askopenfilenames(parent=root,
            filetypes=[("Excel Files", "*.csv"), ("Text Files", "*.txt"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        f=open(filepath)
        f.read()
        f.close()
        
        if not filepath:
            return
        
        dir_ls = filepath
        print(dir_ls)
    
        database=[]
        data_file = dir_ls# "./IN_SITU/"+str(dir_ls)+"_40kV_30uA_NoFilter_Air-001.csv"
        data_p_values = pd.read_csv(data_file,delimiter=',', names=['Channel', 'Intensity'], skiprows = 21)

        de = 20.05*(10**-3)
    
        channel_p = data_p_values.Channel*de
        energy_p = (channel_p[channel_p<=24.0])
        intensity_p = (data_p_values.Intensity[channel_p<=24.0])
        database.append([energy_p, intensity_p])
        
        return database
