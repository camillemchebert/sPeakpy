from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
import specPeak.Preprocessing as Preprocessing
import specPeak.Segmentation as Segmentation
#import specPeak.Data as data
import numpy as np

class Classification:
    """ Classification class """

    def __init__(self, index_segment_, threshold):

        self.energy = Preprocessing.Preprocessing.energy
        self.signal_ = Preprocessing.Preprocessing.signal_
        self.index_segment_=index_segment_
        self.threshold = threshold
        print('Threshold: %d',self.threshold)
        
        self.index_segment_bin = []
        self.index_temp_bin = []
        Classification.percent = []
        self.qualification_=self.qualification()
        self.quantification_=self.quantification()
       
    def reference_clustering(self, x_ref, y_ref, pred):
                
        sigma = (np.std(y_ref))*np.sqrt(np.std(x_ref)**2+np.std(y_ref)**2)
        y_ref_g=sigma*(np.exp(-(x_ref**2)/(sigma**2))*(-x_ref)/(sigma**2))
        
        XX=np.c_[x_ref, y_ref_g]
        
        clustering = SpectralClustering(n_clusters=2,
        assign_labels='kmeans',
        affinity= 'laplacian',
        gamma = .2,
        random_state=None).fit(XX)
           
        sim_y_pred = clustering.fit_predict(XX)    
        eval_score = normalized_mutual_info_score(sim_y_pred, pred)
            
        return eval_score
        
    def qualification(self):
    
        for ind in self.index_segment_:
            
                energy_bin=np.array(self.energy[ind])
                sobel_bin=self.signal_[ind] 
                if (max((sobel_bin))-min((sobel_bin))) !=0:
                    y = (sobel_bin-np.median(sobel_bin))/(max((sobel_bin))-min((sobel_bin)))#(sobel_bin)/(max((sobel_bin))-min((sobel_bin)))#/max(abs(sobel_bin))#/np.std(sobel_bin[0])#(sobel_bin[0]-np.mean(sobel_bin[0]))/np.mean(sobel_bin[0]) #NORMALIZATION
                    x = (energy_bin-np.median(energy_bin))/max((energy_bin)-np.median(energy_bin))#(sum((y*2)*energy_bin)/sum((y*2))) 
                
                X=np.c_[x, y]
                clustering = SpectralClustering(n_clusters=2,
                assign_labels='kmeans',
                affinity= 'laplacian',
                gamma = .2,
                random_state=None).fit(X)
                     
                y_pred = clustering.fit_predict(X)
                score = self.reference_clustering(X[:,0], X[:,1], y_pred)
               
                if np.rint(score*100) >= self.threshold:
                    self.index_temp_bin.append(ind)
                else:
                    Segmentation.Segmentation.index_noise_bin.append(ind)
                    
    def quantification(self):
        noise_mean_val=[]
        for i in Segmentation.Segmentation.index_noise_bin:
            noise_mean_val.append(np.mean(abs(self.signal_[i])))
        
                
        for i in self.index_temp_bin:
            if (np.mean(abs(self.signal_[i])))>np.mean(noise_mean_val):
                self.index_segment_bin.append(i)
                Classification.percent.append((((np.mean(abs(self.signal_[i])))
                                                -np.mean(noise_mean_val))/(np.mean(abs(self.signal_[i]))))*100)
            else:
                Segmentation.Segmentation.index_noise_bin.append(i)
