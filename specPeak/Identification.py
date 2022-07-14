import specPeak.Preprocessing as Preprocessing
import specPeak.Classification as Classification
import numpy as np
import pkgutil
import pandas as pd


class Identification:
    def __init__(self, cs_array, DataType, percent):
       
        self.array = cs_array
        self.percent = percent
        
        if DataType == 'EDXRF':
            self.EDXRF_ID()
            
    def scatter_ID(self, index_segments):

        rayleigh_range=[20., 20.75]
        compton_range=[18.75, 19.5]

        r2 = [22.25, 23]
        c2= [21, 22]

        compton_cs = 0
        rayleigh_cs = 0
        compton_cs2 = 0
        rayleigh_cs2 = 0
        
        for i in index_segments:
            max_ind = np.argmax(Preprocessing.Preprocessing.intensity[i])            
            val = Preprocessing.Preprocessing.energy[i][max_ind]
            
            if val >= compton_range[0] and val <= compton_range[1]:
                compton_cs = [Preprocessing.Preprocessing.energy[i], Preprocessing.Preprocessing.intensity[i]]
                max_val_compton = Preprocessing.Preprocessing.intensity[i][max_ind]
          
            
            if val >= rayleigh_range[0] and val <= rayleigh_range[1]:
                rayleigh_cs = [Preprocessing.Preprocessing.energy[i], Preprocessing.Preprocessing.intensity[i]]
                max_val_rayleigh = Preprocessing.Preprocessing.intensity[i][max_ind]

            if val >= c2[0] and val <= c2[1]:
                compton_cs2 = [Preprocessing.Preprocessing.energy[i], Preprocessing.Preprocessing.intensity[i]]
                max_val_compton2 = Preprocessing.Preprocessing.intensity[i][max_ind]
          
            
            if val >= r2[0] and val <= r2[1]:
                rayleigh_cs2 = [Preprocessing.Preprocessing.energy[i], Preprocessing.Preprocessing.intensity[i]]
                max_val_rayleigh2 = Preprocessing.Preprocessing.intensity[i][max_ind] 

        if compton_cs==0:
            compton_cs = 'None'
        if rayleigh_cs==0:
            rayleigh_cs = 'None'
                    
        print(compton_cs, rayleigh_cs, compton_cs2, rayleigh_cs2)
        return compton_cs, rayleigh_cs, compton_cs2, rayleigh_cs2
            
            
    def EDXRF_ID(self):

        # =============================================================================
        #         ENEGY DISPERSIVE X-RAY FLUORESCENCE IDENTIFICATION
        # =============================================================================
            threshold_energy = 1.2 #keV
            target_energy = 20.0 #keV
            de = 0.020 #keV

            #print((self.array))
            self.cs = self.scatter_ID(self.array)

            energy_Ka1=[]
            energy_Kb1=[]
            energy_La1=[]
            energy_Lb1=[]
            element_list=[]

            data = pkgutil.get_data(__name__, "reference_data/EDXRF.dat")
        
            data = data.split(b'\r\n')
            for line in data[1:]:
                line = line.decode("utf-8").split(',')
                
                element_list.append(str(line[1]))
                energy_Ka1.append(float(line[2]))
                energy_Kb1.append(float(line[4]))
                energy_La1.append(float(line[5]))
                energy_Lb1.append(float(line[7]))
                    
            # =============================================================================
            # Value grid
            # Ka, kb, La, Lb
            #[0], [1], [2], [3]
            # =============================================================================

            
            values_e = np.zeros((len(element_list), 4))
            values_i = np.zeros((len(element_list), 4))
            final_percent = np.zeros((len(element_list)))
            
            
            ID_K = [[False]*len(energy_Ka1),[False]*len(energy_Ka1)]
            ID_L = [[False]*len(energy_Ka1),[False]*len(energy_Ka1)]
            
            max_energy_value=[]
            max_intensity_value=[]
            
            percent = self.percent

            
            for i in self.array:
                max_ind = np.argmax(Preprocessing.Preprocessing.intensity[i])
                max_energy_value.append(Preprocessing.Preprocessing.energy[i][max_ind])#np.mean(Preprocessing.Preprocessing.energy[i]))#, Preprocessing.Preprocessing.energy[i[max_ind]], Preprocessing.Preprocessing.energy[i[max_ind+1]]]))
                max_intensity_value.append(Preprocessing.Preprocessing.intensity[i[max_ind]])

                

##            for r in np.arange(1, 5):
##                for ind in np.arange(len(energy_Ka1)):
##                    for i in np.arange(len(max_energy_value)):
##                        if max_energy_value[i]<target_energy :
##            
##            
##            ###############################     K-edge      ##############################
##                           # recessive_cond = (max_intensity_value[i]<values_i[ind, 0])
##                        
##                            if ID_K[0][ind]==False and abs(max_energy_value[i]-energy_Ka1[ind])<=r*de:
##                                print([element_list[ind], 'Ka1', energy_Ka1[ind],
##                                        max_energy_value[i],
##                                        max_intensity_value[i], percent[i]])
##                                values_e[ind, 1], values_i[ind, 1], final_percent[ind] = max_energy_value[i], max_intensity_value[i], percent[i]
##                                ID_K[0][ind]=True
##                                max_energy_value[i]=np.nan
##                                
##                            if ID_K[0][ind]==True and ID_K[1][ind]==False and values_i[ind, 0]> max_intensity_value[i] and abs(max_energy_value[i]-energy_Kb1[ind])<=r*de:
##                                print([element_list[ind], 'Kb1', energy_Kb1[ind],
##                                        max_energy_value[i], 
##                                        max_intensity_value[i], percent[i]]) 
##                                values_e[ind, 1], values_i[ind, 1], final_percent[ind] = max_energy_value[i], max_intensity_value[i], percent[i]
##                                ID_K[1][ind]=True
##                                max_energy_value[i]=np.nan
##
            sum_peak=[]
            escape_peak=[]
              
            
            for ind in np.arange(len(energy_Ka1)):
                for r in np.arange(1, 3):
                    for i in np.arange(len(max_energy_value)):
                        if max_energy_value[i]<target_energy and max_energy_value[i]>threshold_energy:

                            #############################    Error peaks      ############################ 
            # =============================================================================
            #               Sum peaks
            # =============================================================================

                            if ID_K[0][ind]==True and values_i[ind, 1]>max_intensity_value[i] and (abs((max_energy_value[i])-(2*energy_Ka1[ind])))<=r*de:
                                #print('2*ka1')
                                #print(max_energy_value[i])
                                sum_peak.append([element_list[ind], max_energy_value[i],max_intensity_value[i]])
                                max_energy_value[i]=np.nan
                                #values_e[ind, 1], values_i[ind, 1],final_percent[ind] = max_energy_value[i], 10000000, 0

                            if ID_K[0][ind]==True and ID_K[1][ind]==True and values_i[ind, 1]>max_intensity_value[i] and (abs((max_energy_value[i])-(2*energy_Kb1[ind])))<=r*de:
                                #print('2*kb1')
                                #print(max_energy_value[i])
                                sum_peak.append([element_list[ind], max_energy_value[i],max_intensity_value[i]])
                                max_energy_value[i]=np.nan
                                #values_e[ind, 1], values_i[ind, 1],final_percent[ind] = max_energy_value[i], 10000000, 0
                                
                            if ID_K[0][ind]==True and ID_K[1][ind]==True and values_i[ind, 1]>max_intensity_value[i] and (abs(max_energy_value[i]-(energy_Ka1[ind]+energy_Kb1[ind])))<=r*de:
                                #print('ka1+kb1')
                                #print(max_energy_value[i])
                                sum_peak.append([element_list[ind], max_energy_value[i],max_intensity_value[i]])
                                max_energy_value[i]=np.nan
                                #values_e[ind, 1], values_i[ind, 1],final_percent[ind] = max_energy_value[i], 10000, 0
                #############################    Error peaks      ############################ 
            # =============================================================================
            #               Escape peaks          
            # =============================================================================
                            if ID_K[0][ind]==True and values_i[ind, 1]>max_intensity_value[i] and (abs((max_energy_value[i])-(0.6*energy_Ka1[ind])))<=r*de:
                                #print('2*ka1')
                                #print(max_energy_value[i])
                                escape_peak.append([element_list[ind], max_energy_value[i],max_intensity_value[i]])
                                max_energy_value[i]=np.nan
                                #values_e[ind, 1], values_i[ind, 1],final_percent[ind] = max_energy_value[i], 10000000, 0

                            if ID_K[1][ind]==True and values_i[ind, 1]>max_intensity_value[i] and (abs((max_energy_value[i])-(0.6*energy_Kb1[ind])))<=r*de:
                                #print('2*ka1')
                                #print(max_energy_value[i])
                                escape_peak.append([element_list[ind], max_energy_value[i],max_intensity_value[i]])
                                max_energy_value[i]=np.nan

                            if ID_K[0][ind]==True and values_i[ind, 1]>max_intensity_value[i] and (abs((max_energy_value[i]-(energy_Ka1[ind]))+1.73))<=r*de:
                                #print('2*ka1')
                                #print(max_energy_value[i])
                                escape_peak.append([element_list[ind], max_energy_value[i],max_intensity_value[i]])
                                max_energy_value[i]=np.nan
                                #values_e[ind, 1], values_i[ind, 1],final_percent[ind] = max_energy_value[i], 10000000, 0

                            if ID_K[1][ind]==True and values_i[ind, 1]>max_intensity_value[i] and (abs((max_energy_value[i]-energy_Kb1[ind])+1.73))<=r*de:
                                #print('2*ka1')
                                #print(max_energy_value[i])
                                escape_peak.append([element_list[ind], max_energy_value[i],max_intensity_value[i]])
                                max_energy_value[i]=np.nan
                                #values_e[ind, 1], values_i[ind, 1],final_percent[ind] = max_energy_value[i], 10000000, 0
                
                        
            ###############################     K-edge      ##############################
                            recessive_cond = (max_intensity_value[i]<values_i[ind, 0])
                            error_cond_ka = abs(max_energy_value[i]-energy_Ka1[ind])<=r*de

                            
                                
                            if ID_K[0][ind]==False and error_cond_ka:
                                values_e[ind, 0], values_i[ind, 0], final_percent[ind] = max_energy_value[i], max_intensity_value[i], percent[i]
                                ID_K[0][ind]=True
                                max_energy_value[i]=np.nan
                                
                            if ID_K[0][ind]==True and ID_K[1][ind]==False and values_i[ind, 0]>max_intensity_value[i] and abs(max_energy_value[i]-energy_Kb1[ind])<=r*de:
                                values_e[ind, 1], values_i[ind, 1],final_percent[ind] = max_energy_value[i], max_intensity_value[i], np.mean([percent[i],final_percent[ind],1])
                                ID_K[1][ind]=True
                                max_energy_value[i]=np.nan
                                

                            

                            
                                
             ###############################     L-edge      ##############################        
                            L_a_cond = (ID_L[0][ind]==False and 
                                           abs(max_energy_value[i]-energy_La1[ind])<=r*de)

                            recessive_cond2 = (max_intensity_value[i]<values_i[ind, 2])
                            if L_a_cond and ID_K[0][ind]==False:
                               
                                values_e[ind, 2], values_i[ind, 2], final_percent[ind] = max_energy_value[i], max_intensity_value[i], percent[i]
                                ID_L[0][ind]=True
                                max_energy_value[i]=np.nan
                                
                            if L_a_cond and ID_K[0][ind]==True and recessive_cond:
                                values_e[ind, 2], values_i[ind, 2], final_percent[ind] = max_energy_value[i], max_intensity_value[i], np.mean([percent[i],final_percent[ind],1,1])
                                ID_L[0][ind]=True
                                max_energy_value[i]=np.nan
                                
                            if (ID_L[0][ind]==True and ID_L[1][ind]==False and 
                                values_i[ind, 2]> max_intensity_value[i] and
                                abs(max_energy_value[i]-energy_Lb1[ind])<=r*de):
                                values_e[ind, 3], values_i[ind, 3], final_percent[ind] = max_energy_value[i], max_intensity_value[i], np.mean([percent[i],final_percent[ind],1,1,1])
                                ID_L[1][ind]=True
                                max_energy_value[i]=np.nan
            

                    
                                
            print(sum_peak)
            print(escape_peak)
                            


                            
            
            

                            
                                            
                                        # if abs(max_energy_value[i]-energy_La1[ind])<=r*de:
                                        #     print([element_list[ind], 'La1', energy_La1[ind],
                                        #            max_energy_value[i], 
                                        #            max_intensity_value[i], percent[i]])
                                        #     max_energy_value[i]=-10
                                
                            
                                        #     if abs(max_energy_value[i]-energy_Lb1[ind])<=r*de:
                                        #         print([element_list[ind], 'Lb1',energy_Lb1[ind],
                                        #                max_energy_value[i], 
                                        #                max_intensity_value[i], percent[i]])
                                        #         max_energy_value[i]=-10
                        
            
           
            #   NEED TO BE INTEGRATED
            
            
            Element_ref =pd.DataFrame(
                {
                  "Element": element_list,
                  "Energy Ka": values_e[:, 0], 
                  "Intensity Ka": values_i[:, 0], 
                  "Energy Kb": values_e[:, 1], 
                  "Intensity Kb": values_i[:, 1],
                  "Energy La": values_e[:, 2], 
                  "Intensity La": values_i[:, 2],
                  "Energy Lb": values_e[:, 3], 
                  "Intensity Lb": values_i[:, 3],
                  "Score": final_percent
                  }
                )
            
                      
            Element_selector = Element_ref.loc[
                (Element_ref.loc[:, Element_ref.columns != 'Element']
                  != 0).any(axis=1)]
            
            
            Element_selector = (Element_selector.round(3))
            
            for i in Element_selector:
                Element_selector[i].replace(0, '')  #[Element_selector[i].eq(0)]=''
            
            return Element_selector

    
