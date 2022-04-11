import specPeak.Preprocessing as Preprocessing
import specPeak.Classification as Classification
import os
import numpy as np
import pandas as pd
import pkgutil



class Element_ID:
    def __init__(self, cs_array, DataType):
       
        self.array = cs_array
        self.percent = Classification.Classification.percent
        if DataType == 'EDXRF':
            self.EDXRF_ID()
            
            
    def EDXRF_ID(self):        
        
            tube_energy = 20. #keV
            de = 0.0201
            ###########################       XRF DATASET      ###########################

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
                max_energy_value.append(Preprocessing.Preprocessing.energy[i[max_ind]])
                max_intensity_value.append(Preprocessing.Preprocessing.intensity[i[max_ind]])

            print(max_energy_value)
                
                                    
##            for r in np.arange(1, 5):
##                for ind in np.arange(len(energy_Ka1)):
##                    for i in np.arange(len(max_energy_value)):
##                        if max_energy_value[i]<tube_energy :
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
              
            for r in np.arange(1, 6):
                for ind in np.arange(len(energy_Ka1)):
                    for i in np.arange(len(max_energy_value)):
                        if max_energy_value[i]<tube_energy :


            ###############################     K-edge      ##############################
                            recessive_cond = (max_intensity_value[i]<values_i[ind, 0])
                            error_cond_ka = abs(max_energy_value[i]-energy_Ka1[ind])<=r*de
                        
                            if ID_K[0][ind]==False and error_cond_ka:
                                print([element_list[ind], 'Ka1', energy_Ka1[ind],
                                       max_energy_value[i],
                                       max_intensity_value[i], percent[i]])
                                values_e[ind, 0], values_i[ind, 0] = max_energy_value[i], max_intensity_value[i]
                                ID_K[0][ind]=True
                                max_energy_value[i]=np.nan
                                
                            if ID_K[0][ind]==True and ID_K[1][ind]==False and values_i[ind, 0]> max_intensity_value[i] and abs(max_energy_value[i]-energy_Kb1[ind])<=r*de:
                                print([element_list[ind], 'Kb1', energy_Kb1[ind],
                                       max_energy_value[i], 
                                       max_intensity_value[i], percent[i]]) 
                                values_e[ind, 1], values_i[ind, 1] = max_energy_value[i], max_intensity_value[i]
                                ID_K[1][ind]=True
                                max_energy_value[i]=np.nan
                                
                            L_a_cond = (ID_L[0][ind]==False and 
                                           abs(max_energy_value[i]-energy_La1[ind])<=r*de)
                            
                            if L_a_cond and ID_K[0][ind]==False:
                                print([element_list[ind], 'La1', energy_La1[ind],
                                       max_energy_value[i],
                                       max_intensity_value[i], percent[i]])
                                values_e[ind, 2], values_i[ind, 2] = max_energy_value[i], max_intensity_value[i]
                                ID_L[0][ind]=True
                                max_energy_value[i]=np.nan
                                
                            if L_a_cond and ID_K[0][ind]==True  and ID_K[0][ind]==True and recessive_cond:
                                values_e[ind, 2], values_i[ind, 2] = max_energy_value[i], max_intensity_value[i]
                                ID_L[0][ind]=True
                                max_energy_value[i]=np.nan
                                
                            if (ID_L[0][ind]==True and ID_L[1][ind]==False and 
                                values_i[ind, 2]> max_intensity_value[i] and
                            abs(max_energy_value[i]-energy_Lb1[ind])<=r*de):
                                values_e[ind, 3], values_i[ind, 3] = max_energy_value[i], max_intensity_value[i]
                                ID_L[1][ind]=True
                                max_energy_value[i]=np.nan
                                            
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
                        
            
            #############################    Error peaks      ############################ 
            # =============================================================================
            #       Escape peaks
                                
            # =============================================================================
            
            
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
                  "Percentage": final_percent
                  }
                )
            
            Element_selector = Element_ref.loc[
                (Element_ref.loc[:, Element_ref.columns != 'Element']
                  != 0).any(axis=1)]
            
            
            Element_selector = (Element_selector.round(3))
            
            for i in Element_selector:
                Element_selector[i].replace(0, '')  #[Element_selector[i].eq(0)]=''
            
            return Element_selector
