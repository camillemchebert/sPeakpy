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
import random

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
        energy_p = channel_p[channel_p<=39.0]
        intensity_p = data_p_values.Intensity[channel_p<=39.0]
        database.append([energy_p, intensity_p])
        
    return database


    

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz
import math
import pylab as plb
from scipy.special import erf


# Synthetic spectra

channel=0.0208
channels=np.arange(-1, 40, channel)

energy_Ka1=[]
energy_Kb1=[]
energy_La1=[]
energy_Lb1=[]
element_list=[]

# with open("Energy_Levels.txt", "r") as file:
#     data = file.readlines()[1:]
#     for line in data:
#         line = line.split(',')
#         element_list.append(str(line[1]))
#         energy_Ka1.append(float(line[2]))
#         energy_Kb1.append(float(line[4]))
#         if np.isnan(float(line[5])) or np.isnan(float(line[7])):
#             energy_La1.append(0)
#             energy_Lb1.append(0)
            
#         else:
#             energy_La1.append(float(line[5]))
#             energy_Lb1.append(float(line[7]))
        


input_data = import_data()

for i in np.arange(np.shape(input_data)[0]):
    energy = np.array(input_data[i][0]) 
    intensity = np.array(input_data[i][1])
    element_peaks = sp.Peak(energy, intensity, p_res=150, b_res=20)
    print((max(element_peaks.e.cs[1][1])/sum(element_peaks.e.cs[0][1])))#-((sum(element_peaks.e.cs[3][1])/sum(element_peaks.e.cs[2][1]))-((sum(element_peaks.e.cs[1][1])/sum(element_peaks.e.cs[0][1])))))
    print(((max(element_peaks.e.cs[1][1])/sum(element_peaks.e.cs[0][1])))+(((sum(element_peaks.e.cs[3][1])/sum(element_peaks.e.cs[2][1]))-(sum(element_peaks.e.cs[1][1])/sum(element_peaks.e.cs[0][1])))))
    print(((max(element_peaks.e.cs[1][1])/sum(element_peaks.e.cs[0][1])))+(((max(element_peaks.e.cs[3][1])/sum(element_peaks.e.cs[2][1])))))#-(sum(element_peaks.e.cs[1][1])/sum(element_peaks.e.cs[0][1])))))
    
    print('--------------------------------------')
    

# energy_MA =[]
# MA = []

# with open("specPeak/reference_data/Rh_MA.txt", "r") as file:
#     data = file.readlines()
#     for line in data:
#         line = line.split()
#         energy_MA.append(float(line[0]))
#         MA.append(float(line[1]))

# def brem(E, a, b, c, B):
#     """
#     .......................

#     """
#     E = E/1000
#     E0 = .04
#     k=1/E
    
    
#     MA_array=[]
#     E_array =[]
  
#     for i in np.arange(len(energy_MA)-1):
#         for j in np.arange(len(E)):
#             if i==0:
#                 if energy_MA[0]>E[j]:
#                     MA_array.append(MA[i])
#             if energy_MA[i]<=E[j]:
#                 if energy_MA[i+1]>=E[j]:
#                     MA_array.append(MA[i])

#     MA_array = np.array(MA_array)
   
#     brem_dis = np.exp((-c*E))*(1/E)*(a-(b*np.log(1/E)))*(1-B*(E/E0))
    
#     return brem_dis
    
# def V(x, a, alpha, gamma):
#     """
#     Return the Voigt line shape at x with Lorentzian component HWHM gamma
#     and Gaussian component HWHM alpha.

#     """
#     sigma = alpha / np.sqrt(2 * np.log(2))
#     x=x-np.mean(x)
#     dis = a*(np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma )/np.sqrt(2*np.pi)
#     #skew = (1 + erf((e * (x)) / (a * np.sqrt(2))))
#     return dis#*skew

# # def stacked_spectrum(energy, intensity, energy_array, intensity_array):
# #     sorted_energy_array=[]
# #     sorted_intensity_array=[]
# #     if len(energy)!=0:
# #         for i in np.arange(len(energy_array)):
# #             for j in np.arange(len(sorted_energy_array)):
# #                 if energy_array[i]==sorted_energy_array[j]:
# #                     intensity_array[i]=(sorted_intensity_array[j]+intensity_array[i])
# #                     energy_array[i]==sorted_energy_array[j]
# #                 else:
# #                     intensity_array.append(intensity_array[j])
# #                     energy_array.append(energy_array[j])
                    
# #     return sorted_energy_array, sorted_intensity_array



# def stacked_spectrum(energy_array, intensity_array, referenced_array):
   
#     ele=np.shape(energy_array)[0]
    
#     length=1
    
#     stacked_intensity_array=np.zeros(len(referenced_array))
#     if length!=0:
#         for i in np.arange(ele):
#             length=len(energy_array[i])
#             for j in np.arange(length):
#                 for k in np.arange(len(stacked_intensity_array)):
#                     if energy_array[i][j]==referenced_array[k]:
#                         stacked_intensity_array[k]=(stacked_intensity_array[k]+intensity_array[i][j])
                    
#     else:
#         print('Unexpected Error: stacked_spectrum() requires 2 array arguments')
#         print('Array arguments empty')
        
#     return stacked_intensity_array


# def element_ID(elements, max_energy):
#     energy_Ka = []; energy_Kb = [];
#     energy_La = []; energy_Lb = []; intensity_Ka = []; intensity_Kb = []
#     selected_elements= []
#     for i in np.arange(len(elements)):
#         for j in np.arange(len(element_list)):
#             if elements[i] == element_list[j]:
#                 if energy_Ka1[j]<= max_energy or energy_La1[j]<= max_energy or energy_Kb1[j]<= max_energy or energy_Lb1[j]<= max_energy:
#                     ind1=np.where(abs(energy_Ka1[j]-channels)==min(abs(energy_Ka1[j]-channels)))[0][0]
#                     energy_1=channels[ind1]
#                     ind2=np.where(abs(energy_Kb1[j]-channels)==min(abs(energy_Kb1[j]-channels)))[0][0]
#                     energy_2=channels[ind2]
                    
#                     ind3=np.where(abs(energy_La1[j]-channels)==min(abs(energy_La1[j]-channels)))[0][0]
#                     energy_3=channels[ind3]
#                     ind4=np.where(abs(energy_Lb1[j]-channels)==min(abs(energy_Lb1[j]-channels)))[0][0]
#                     energy_4=channels[ind4]
                    
#                     energy_Ka.append(energy_1)
#                     energy_Kb.append(energy_2)
#                     energy_La.append(energy_3)
#                     energy_Lb.append(energy_4)
                    
#                     max_random_value=45000
                    
                    
                        
                    
                    
#                     random_intensity=np.random.randint(0,  max_random_value)
                    
#                     Ka = random_intensity
#                     Kb = random_intensity*np.random.rand()
                    
#                     intensity_Ka.append(Ka)
#                     intensity_Kb.append(Kb)
#                     selected_elements.append(elements[i]+' Ka')
#                     selected_elements.append(elements[i]+' Kb')
                
#     return energy_Ka, energy_Kb, energy_La, energy_Lb, selected_elements

# ##############################################################################
# ##############################################################################
# ###########################   Spectral lines

# def simulated_lines(min_energy, dt, num):
#     energy_array = np.arange(min_energy, min_energy+dt*(num), dt)
#     intensity_array = brem(np.arange(0.3,40, 0.1),*np.array([-3.21791701, -0.40700151, 34.89965323]))
    
#     plt.figure()
#     fig, ax = plt.subplots(figsize=(12, 6))

#     plt.plot(energy_array, intensity_array, 'o')
#     ax.vlines([energy_array], 0, intensity_array, linestyles='dashed', colors='lightgray')
#     plt.show()
    
#   #  cross_section_Ka1=[channels[ind1-20:ind1+21],np.convolve(random_intensity, V(channels[ind1-20:ind1+21], .5, .05, .05))/max(V(channels[ind1-20:ind1+21], .5, .05, .05))]
    
#     return energy_array, intensity_array
    


# def simulated_spectral_lines(elements, max_energy, channels):
#     # element randomized if list is empty

#     energy_array=[]
#     intensity_array=[]
    
#     plt.figure(1)
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     #compton=[np.arange(17.9, 18.5, 0.01),np.convolve(200, V(np.arange(17.9, 18.5, 0.01),1 , 0.2, 0.2))]
#     #cross_section_Kb1=[channels[ind2-83:ind2+84],np.convolve(beta, V(channels[ind2-83:ind2+84], 1,sigma*0.9, sigma*0.9))]
#     #energy_array.append(compton[0])
#     #intensity_array.append(compton[1])
    
#     for i in np.arange(len(elements)):
#         for j in np.arange(len(element_list)):
#             if elements[i] == element_list[j]:
#                 if energy_Ka1[j]<= max_energy or energy_La1[j]<= max_energy or energy_Kb1[j]<= max_energy or energy_Lb1[j]<= max_energy:
                
#                     #print(elements[i])
#                     max_random_value=5e4
#                     sigma=(0.05-0.03)*np.random.rand() + 0.03
#                     random_intensity=np.random.randint(0,  max_random_value)
                    
#                     alpha = random_intensity
#                     beta = random_intensity*((0.6-0.4)*np.random.rand() + 0.4)
                    
#                     small_intensity = 6.
                    
                    
                    
#                     if energy_Ka1[j]< small_intensity or energy_La1[j]< small_intensity or energy_Kb1[j]< small_intensity or energy_Lb1[j]< small_intensity:
#                         random_intensity=random_intensity*0.2
                        
#                     alpha = random_intensity
#                     beta = random_intensity*((0.6-0.4)*np.random.rand() + 0.4)
                        
#                     if element_list[j] == 'Rh':
#                         random_intensity=250
#                         sigma=0.1
#                         alpha = random_intensity
#                         beta = random_intensity*((0.5-0.2)*np.random.rand() + 0.2)
                        
                    
#                     #print(random_intensity)
                        
                    
                    
                    
                    
                
                    
                    
                    
#                     ind1=np.where(abs(energy_Ka1[j]-channels)==min(abs(energy_Ka1[j]-channels)))[0][0]
#                     energy_1=channels[ind1]
#                     ind2=np.where(abs(energy_Kb1[j]-channels)==min(abs(energy_Kb1[j]-channels)))[0][0]
#                     energy_2=channels[ind2]
#                     ind3=np.where(abs(energy_La1[j]-channels)==min(abs(energy_La1[j]-channels)))[0][0]
#                     energy_3=channels[ind3]
#                     ind4=np.where(abs(energy_Lb1[j]-channels)==min(abs(energy_Lb1[j]-channels)))[0][0]
#                     energy_4=channels[ind4]
                 
#                     cross_section_Ka1=[channels[ind1-83:ind1+84],np.convolve(alpha, V(channels[ind1-83:ind1+84],1 , sigma, sigma))]
#                     cross_section_Kb1=[channels[ind2-83:ind2+84],np.convolve(beta, V(channels[ind2-83:ind2+84], 1,sigma*0.9, sigma*0.9))]
                    
#                       # Convolution Voigt and spectral lines
#                     #plt.plot(cross_section_Ka1[0],cross_section_Ka1[1])
#                     plt.text(energy_1, random_intensity+(max_random_value*0.02), (elements[i]+'$_K$'+'$_\u03B1$'))
    
#                     #plt.plot(cross_section_Kb1[0],cross_section_Kb1[1])
#                     plt.text(energy_2, random_intensity*0.3+(max_random_value*0.02), (elements[i]+'$_K$'+'$_\u03B2$'))
#                     #print(cross_section_Ka1[0])
#                 # K alpha lines
#                     plt.plot(energy_1, random_intensity, 'o')
#                     ax.vlines([energy_1], 0, random_intensity, linestyles='dashed', colors='lightgray')
                    
                    
#                     # K beta lines
#                     plt.plot(energy_2, random_intensity*0.3, 'o')
#                     ax.vlines([energy_2], 0, random_intensity*0.3, linestyles='dashed', colors='lightgray')
                    
#                     energy_array.append(cross_section_Ka1[0])
#                     energy_array.append(cross_section_Kb1[0])
                    
#                     intensity_array.append(cross_section_Ka1[1])
#                     intensity_array.append(cross_section_Kb1[1])
                        
#                     # if ind3 >=21 or ind4 >=21:
                        
#                     #     cross_section_La1=[channels[ind3-20:ind3+21],np.convolve(alpha*0.2, V(channels[ind3-20:ind3+21], 1, sigma, sigma))/max(V(channels[ind3-20:ind3+21], 1, sigma, sigma))]
#                     #     cross_section_Lb1=[channels[ind4-20:ind4+21],np.convolve(beta*0.1, V(channels[ind4-20:ind4+21], 1, sigma, sigma))/max(V(channels[ind4-20:ind4+21], 1, sigma*0.9, sigma*0.9))]
        
                       
                        
                        
#                     #     # L alpha lines
#                     #     plt.plot(energy_3, random_intensity*0.2, 'o')
#                     #     ax.vlines([energy_3], 0, random_intensity*0.2, linestyles='dashed', colors='lightgray')
                        
                        
#                     #     # L beta lines
#                     #     plt.plot(energy_4, random_intensity*0.1, 'o')
#                     #     ax.vlines([energy_4], 0, random_intensity*0.1, linestyles='dashed', colors='lightgray')
                        
                    
                       
        
#                     #     plt.text(energy_3, random_intensity*0.2+(max_random_value*0.02), (elements[i]+'$_L$'+'$_\u03B1$'))
        
#                     #     #plt.plot(cross_section_Kb1[0],cross_section_Kb1[1])
#                     #     plt.text(energy_4, random_intensity*0.1+(max_random_value*0.02), (elements[i]+'$_L$'+'$_\u03B2$'))
        
                        
                      
#                     #     energy_array.append(cross_section_La1[0])
#                     #     energy_array.append(cross_section_Lb1[0])
                        
                        
#                     #     intensity_array.append(cross_section_La1[1])
#                     #     intensity_array.append(cross_section_Lb1[1])
        
#                 #ene, inte = stacked_spectrum(cross_section_Ka1[0],cross_section_Ka1[1], energy_array, intensity_array)
        
    
    
#     # Add baseline
# #    baseline=          
               
#     plt.title('A')
#     plt.ylim(0, 500); plt.xlim(0, max(channels))
#     #plt.title('Spectral lines')
#     #plt.xlabel('Energy (kV')
#     plt.ylabel('Intensity (cps)')
#     plt.show()
    
#     return energy_array, intensity_array
#   #  return ind1, ind2, Kalpha, Kbeta


    
# # def cross_section_convolution(ind1, ind2, Kalpha, Kbeta, channels):
    
# #     energy_array=[]
# #     intensity_array=[]
    
# #     cross_section_Ka1=[channels[ind1-30:ind1+31],np.convolve(Kalpha, V(channels[ind1-30:ind1+31], .5, .05, .05))/max(V(channels[ind1-30:ind1+31], .5, .05, .05))]
# #     cross_section_Kb1=[channels[ind2-30:ind2+31],np.convolve(Kbeta, V(channels[ind2-30:ind2+31], .5, .05, .05))/max(V(channels[ind2-30:ind2+31], .5, .05, .05))]
    
# #     plt.figure()
# # # Convolution Voigt and spectral lines
# #     plt.plot(cross_section_Ka1[0],cross_section_Ka1[1])
# #     plt.text(channels[ind1], Kalpha+(max_random_value*0.02), str(elements[i]))

# #     plt.plot(cross_section_Kb1[0],cross_section_Kb1[1])
# #     plt.text(energy_2, Kbeta+(max_random_value*0.02), str(elements[i]))
# #                 #print(cross_section_Ka1[0])
# #     plt.show()
                
# #     energy_array.append(cross_section_Ka1[0])
# #     energy_array.append(cross_section_Kb1[0])
    
# #     intensity_array.append(cross_section_Ka1[1])
# #     intensity_array.append(cross_section_Kb1[1])
    
# #     return energy_array, intensity_array
    
    
# ##############################################################################
# ##############################################################################
# ###########################   Calling functions
    
# elements = ['K', 'Si', 'Ca', 'Fe',  'Ni', 'Mn', 'Cu', 'Rh', 'As']


# e, i = simulated_spectral_lines(elements, 40, channels)


# i_brem = brem(np.linspace(1,40, len(channels)),*np.array([-3.13512578, -0.42263127, 119.26479176, 14.0978873]))*10 #-3.21791701, -0.40700151, 34.89965323

# stacked_i=stacked_spectrum(e, i, channels)

# stacked_i += i_brem

# e1, e2, e3, e4, name_list = element_ID(elements, 40)

# noise1 = np.random.uniform(0,.05,len(stacked_i))
# noise2 = np.random.uniform(0,.1,len(stacked_i))
# noise3 = np.random.uniform(0,.2,len(stacked_i))
# noise4 = np.random.uniform(0,.4,len(stacked_i))
# noise5 = np.random.uniform(0,.8,len(stacked_i))

# ##############################################################################
# ##############################################################################

# plt.figure(2)
# fig, ax = plt.subplots(figsize=(12, 6))

# ax.vlines(e1, 0, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e2, 0, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e3, 0, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e4, 0, 500, linestyles='dashed', colors='lightgray')
# plt.plot(channels, stacked_i)

# plt.ylim(0, 500); plt.xlim(0, max(channels))
# plt.title('B')
# plt.ylabel('Intensity (cps)')

# plt.show()

# ##############################################################################
# ##############################################################################

# plt.figure(3)
# fig, ax = plt.subplots(figsize=(12, 6))

# ax.vlines(e1, 0, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e2, 0, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e3, 0, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e4, 0, 500, linestyles='dashed', colors='lightgray')
# plt.plot(channels, stacked_i+noise1)

# plt.title('C')
# plt.ylim(0, 500); plt.xlim(0, max(channels))
# plt.xlabel('Energy (kV')
# plt.ylabel('Intensity (cps)')

# plt.show()

# plt.figure(4)
# fig, ax = plt.subplots(figsize=(12, 3))

# ax.vlines(e1, -10, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e2, -10, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e3, -10, 500, linestyles='dashed', colors='lightgray')
# ax.vlines(e4, -10, 500, linestyles='dashed', colors='lightgray')
# plt.plot(channels, noise1)
# SNR = noise1

# mu = abs(SNR).mean()
# #median = np.median(SNR)
# sigma = abs(SNR).std()
# textstr = '\n'.join((
#     r'$\mu=%.2f$' % (mu, ),
#     r'$\sigma=%.2f$' % (sigma, )))


# # place a text box in upper left in axes coords
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#     verticalalignment='top', bbox=props)

# plt.title('D')
# plt.ylim(min(noise1)*1.2,max(noise1)*1.2); 
# plt.xlim(0, max(channels))
# plt.xlabel('Energy (kV')
# plt.ylabel('Intensity (cps)')

# plt.show()

# ##############################################################################
# ##############################################################################

# plt.figure(5)
# fig, ax = plt.subplots(figsize=(6, 5))
# n, bins, patches = plt.hist(x=noise2, bins='auto', color='gray',
#                             alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Intensity variation')
# plt.ylabel('Frequency')
# plt.title('Noise distribution')
# maxfreq = n.max()

# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.show()


# element_peaks = sp.Peak(channels, stacked_i+noise2*100, p_res=140, b_res=20)
# element_peaks.get_data('sim_noise1')
# element_peaks = sp.Peak(channels, stacked_i+noise2*1000, p_res=140, b_res=20)
# element_peaks.get_data('sim_noise2')
# plt.figure()
# plt.plot(channels, stacked_i+noise5*100)

