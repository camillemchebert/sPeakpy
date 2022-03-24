class Classification(Preprocessing):
    """ Classification class """

    sk = __import__('sklearn')
    sk_m = sk.metrics.cluster._supervised
    sk_c=sk.cluster._spectral
    
    def __init__(self, index_segment_, threshold=70):
        
        self.index_segment_=index_segment_
        self.threshold = threshold
        
        self.index_segment_bin = []
        self.index_temp_bin = []
        
        self.qualification_=self.qualification()
    
        self.quantification_=self.quantification()
       
        
    def reference_clustering(self, x_ref, y_ref, pred):
        #completeness_score #homogeneity_completeness_v_measure #completeness_score
        
        
        sigma = (np.std(x_ref))*np.sqrt(np.std(x_ref)**2+np.std(y_ref)**2)#(np.std(abs(xx))/np.std(abs(yy)))/2#(np.std(abs(xx))/np.std(abs(yy)))*np.sqrt(np.std(xx)**2+np.std(yy)**2)#*np.std(abs(yy))#/np.std(abs(yy))#-(abs(np.mean(xx)+abs(np.mean(yy))))#/np.mean(xx))#-np.std(abs(yy))) # (max(xx)-min(xx))/
        y_ref_g = np.zeros(np.shape(x_ref))
        
        y_ref_g=sigma*(np.exp(-(x_ref**2)/(sigma**2))*(-x_ref)/(sigma**2))
        
        
        XX=np.c_[x_ref, y_ref_g]
        clustering = self.sk_c.SpectralClustering(n_clusters=2,
        assign_labels='kmeans',
        affinity= 'laplacian',
        gamma = .2,
        random_state=None).fit(XX)
           
        sim_y_pred = clustering.fit_predict(XX)
        
        #scaled_diff= 1-np.mean(abs(y_ref-y_ref_g)/sum(abs(y_ref_g)))
        scaled_diff = 1-np.mean(abs(y_ref_g-y_ref))
        label_score = self.sk_m.normalized_mutual_info_score(sim_y_pred, pred)
        eval_score = np.mean([scaled_diff, label_score])
        
        return eval_score
        
    def qualification(self):
    
        max_ci = []
        noise_mean_val = []
        
        self.val_cs = max_ci
        self.val_noise = noise_mean_val
        
        self.threshold=60
        
        for ind in self.index_segment_:
            
            if len(ind)>5:
            
                energy_bin=Preprocessing.energy[ind]
                sobel_bin=Preprocessing.signal_[ind] 
                if (max(abs(sobel_bin))-min(abs(sobel_bin))) !=0:
                
                    y = (sobel_bin-np.median(sobel_bin))/(max(abs(sobel_bin))-min(abs(sobel_bin)))#(sobel_bin)/(max((sobel_bin))-min((sobel_bin)))#/max(abs(sobel_bin))#/np.std(sobel_bin[0])#(sobel_bin[0]-np.mean(sobel_bin[0]))/np.mean(sobel_bin[0]) #NORMALIZATION
                    x = (energy_bin-np.median(energy_bin))/max((energy_bin)-np.median(energy_bin))#(sum((y*2)*energy_bin)/sum((y*2))) 
                else:
                    print(ind)
                
                      
                X=np.c_[x, y]
                clustering = self.sk_c.SpectralClustering(n_clusters=2,
                assign_labels='kmeans',
                affinity= 'laplacian',
                gamma = .2,
                random_state=None).fit(X)
                     
                y_pred = clustering.fit_predict(X)
    
                score = self.reference_clustering(X[:,0], X[:,1], y_pred)
               
                if np.rint(score*100) >= self.threshold:
                    max_ci.append(np.mean(abs(sobel_bin)))
                    self.index_temp_bin.append(ind)
            else:
                noise_mean_val.append(np.mean(abs(sobel_bin)))
                Segmentation.index_noise_bin.append(ind)
                    
         

    def quantification(self):
        
        max_ci=self.val_cs
        noise_mean_val = self.val_noise
        
        for i in np.arange(len(max_ci)):
            if (max_ci[i])>=np.mean(noise_mean_val):
                self.index_segment_bin.append(self.index_temp_bin[i])   
            else:
                Segmentation.index_noise_bin.append(self.index_temp_bin[i])