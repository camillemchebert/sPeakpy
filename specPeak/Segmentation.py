import numpy as np

class Segmentation:
    def __init__(self, signal_):
        self.signal_ = signal_
        
        Segmentation.index_noise_bin = []
    
        crest_ind=[]
        trough_ind=[]
        
        filter_data = np.real(self.signal_)
        
        for i in np.arange(0, len(filter_data)-1, 1):
            
        # =============================================================================
        #         Positive feature cross-over point
        # =============================================================================
            if filter_data[i+1]>=0 and filter_data[i]<=0:
                crest_ind.append(i+np.argmin([abs(filter_data[i]),
                                                    abs(filter_data[i+1])]))
        # =============================================================================
        #         Negative feature cross-over point
        # =============================================================================
            if filter_data[i+1]<=0 and filter_data[i]>=0:
                trough_ind.append(i+np.argmin([abs(filter_data[i]),
                                                      abs(filter_data[i+1])]))
        
                
        ##############################################################################
        ##############################################################################
        ######################         SEGMENTATION 2.0           ####################
        ##############################################################################
        ##############################################################################
        
        # =============================================================================
        #       Equal number of data points around crest index
        # =============================================================================
        
        index_array=[]
        
        sobel_positive_feature = []
        sobel_negative_feature = []
        noise_array=[]
        
        
        for i in np.arange(np.shape(crest_ind)[0]-1):
            for j in np.arange(np.shape(trough_ind)[0]):
                if crest_ind[i]<trough_ind[j] and crest_ind[i+1]>trough_ind[j]:
                    positive_feature = np.arange(crest_ind[i], trough_ind[j]+1)
                    negative_feature = np.arange(trough_ind[j], crest_ind[i+1])
                    if len(positive_feature)>2 and len(negative_feature)>2:
                        if len(positive_feature)>=len(negative_feature):
                            index_array.append(np.arange(trough_ind[j]-len(negative_feature),trough_ind[j]+len(negative_feature)+1))
                                           
                        else:
                            index_array.append(np.arange(trough_ind[j]-len(positive_feature),trough_ind[j]+len(positive_feature)+1))

                    else:
                        Segmentation.index_noise_bin.append(np.arange(crest_ind[i], crest_ind[i+1]))
        
        self.index_segment_=index_array
