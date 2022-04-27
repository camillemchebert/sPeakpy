import numpy as np

class Segmentation:
    def __init__(self, signal_):

        # =============================================================================
        #         Class array
        # =============================================================================
        Segmentation.index_noise_bin = []
        pos_ind=[]
        neg_ind=[]
        index_array=[]
        
        for i in np.arange(0, len(signal_)-1):
        # =============================================================================
        #         Positive feature cross-over point
        # =============================================================================
            if signal_[i+1]>=0 and signal_[i]<=0:
                pos_ind.append(i+np.argmin([abs(signal_[i]),abs(signal_[i+1])]))
        # =============================================================================
        #         Negative feature cross-over point
        # =============================================================================
            elif signal_[i+1]<=0 and signal_[i]>=0:
                neg_ind.append(i+np.argmin([abs(signal_[i]),abs(signal_[i+1])]))
        # =============================================================================
        #       Equal number of data points around crest index
        # =============================================================================
        for i in np.arange(np.shape(pos_ind)[0]-1):
            for j in np.arange(np.shape(neg_ind)[0]):
                if pos_ind[i]<neg_ind[j] and pos_ind[i+1]>neg_ind[j]:
                    positive_feature = np.arange(pos_ind[i], neg_ind[j]+1)
                    negative_feature = np.arange(neg_ind[j], pos_ind[i+1])
                    if len(positive_feature)>=3 and len(negative_feature)>=3:
                        if len(positive_feature)>=len(negative_feature):
                            index_array.append(np.arange(neg_ind[j]-len(negative_feature)+1,
                                                         neg_ind[j]+len(negative_feature)))              
                        else:
                            index_array.append(np.arange(neg_ind[j]-len(positive_feature)+1,
                                                         neg_ind[j]+len(positive_feature)))

                    else:
                        Segmentation.index_noise_bin.append(np.arange(pos_ind[i],
                                                                      pos_ind[i+1]+1))
        
        self.index_segment_=index_array
       
