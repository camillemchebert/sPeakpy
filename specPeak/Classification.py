from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_mutual_info_score
import specPeak.Preprocessing as Preprocessing
import specPeak.Segmentation as Segmentation
#import specPeak.Data as data
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import norm



class Classification:
    """ Classification class """

    def __init__(self, index_segment_, threshold):
        
        if threshold==None:
            self.threshold = 60
        else:
            self.threshold = threshold
            
        self.energy = Preprocessing.Preprocessing.energy
        self.signal_ = Preprocessing.Preprocessing.signal_
        self.index_segment_=index_segment_
        
        print('Threshold: %f' %self.threshold)
        
        self.index_segment_bin = []
        self.index_temp_bin = []
        self.percent = []

        self.qualification_=self.qualification()
        self.quantification_=self.quantification()
       
    def reference_clustering(self, x_ref, y_ref, pred):
                

        sim_y_pred = np.ones(len(y_ref))
        sim_y_pred[0:int(np.rint(len(y_ref)/2))]=0
        eval_score = adjusted_mutual_info_score(sim_y_pred, pred)
            
        return eval_score
        
    def qualification(self):
    
        for ind in self.index_segment_:
                energy_bin=np.array(self.energy[ind])
                sobel_bin=self.signal_[ind] 
                if (max(abs(sobel_bin))-min(abs(sobel_bin))) !=0:
                    y = (sobel_bin-np.mean(sobel_bin))/(max(abs(sobel_bin))-min(abs(sobel_bin)))
                    x = np.arange(len(y))
                
                X=np.c_[x,y]
                clustering = SpectralClustering(n_clusters=2,
                assign_labels='kmeans',
                affinity= 'laplacian',
                random_state=None).fit(X)
                 
                y_pred = clustering.fit_predict(X)
                score = self.reference_clustering(X[:,0], X[:,1], y_pred)
                   
                if score*100 >= self.threshold:
                    self.index_temp_bin.append(ind)
                else:
                    Segmentation.Segmentation.index_noise_bin.append(ind)


    def quantification(self):
        noise_sum_val=[]

        for j in Segmentation.Segmentation.index_noise_bin:
            noise_sum_val.append(sum(abs(self.signal_[j])))

        quantification_criterion=np.mean(noise_sum_val)
        for i in self.index_temp_bin:
            peak_criterion = sum(abs(self.signal_[i]))
            threshold_criterion=((peak_criterion)/(quantification_criterion+peak_criterion))*100
            #print(peak_criterion)
            self.percent.append(threshold_criterion)
            if threshold_criterion>=self.threshold:
                #print(threshold_criterion)
                self.index_segment_bin.append(i)
            else:
                Segmentation.Segmentation.index_noise_bin.append(i)
               # noise_sum_val.append(sum(abs(self.signal_[i])))
