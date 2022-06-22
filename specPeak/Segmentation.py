import numpy as np
import matplotlib.pyplot as plt
import specPeak.Preprocessing as Preprocessing
from matplotlib import ticker


class Segmentation:
    def __init__(self, signal_, plot_):

        self.signal = signal_

        self.pos_bin=[]
        self.neg_bin=[]

        self.seg_()

        if plot_ == True:
            self.plot_data()

        
        
    def seg_(self):
        

        # =============================================================================
        #         Class array
        # =============================================================================
        Segmentation.index_noise_bin = []
        pos_ind=[]
        neg_ind=[]
        index_array=[]
        index_arrayy=[]

        
        for i in np.arange(0, len(self.signal)-1):
        # =============================================================================
        #         Positive feature cross-over point
        # =============================================================================
            if self.signal[i+1]>=0 and self.signal[i]<=0:
                pos_ind.append(i+np.argmin([abs(self.signal[i]),abs(self.signal[i+1])]))
        # =============================================================================
        #         Negative feature cross-over point
        # =============================================================================
            if self.signal[i+1]<=0 and self.signal[i]>=0:
                neg_ind.append(i+np.argmin([abs(self.signal[i]),abs(self.signal[i+1])]))
        # =============================================================================
        #       Equal number of data points around crest index
        # =============================================================================
        for i in np.arange(0, np.shape(pos_ind)[0]-1):
            for j in np.arange(0, np.shape(neg_ind)[0]):
                if pos_ind[i]<neg_ind[j] and pos_ind[i+1]>neg_ind[j]:
                    positive_feature = np.arange(pos_ind[i], neg_ind[j])
                    negative_feature = np.arange(neg_ind[j], pos_ind[i+1])
                    
                    self.pos_bin.append(positive_feature)
                    self.neg_bin.append(negative_feature)
                    #

                    #np.arctan(self.signal[np.arange(pos_ind[i], pos_ind[i+1])]/x[np.arange(pos_ind[i], pos_ind[i+1])])
                    
                    if len(positive_feature)>3 and len(negative_feature)>3:
                        index_arrayy.append(np.arange(pos_ind[i], pos_ind[i+1]+1))
                        
                        
                        if len(positive_feature)>=len(negative_feature):
                            index_array.append(np.arange(neg_ind[j]-len(negative_feature),
                                                         neg_ind[j]+len(negative_feature)+1))              
                        if len(positive_feature)<len(negative_feature):
                            index_array.append(np.arange(neg_ind[j]-len(positive_feature),
                                                         neg_ind[j]+len(positive_feature)+1))
                    else:
                        Segmentation.index_noise_bin.append(np.arange(pos_ind[i],
                                                                      pos_ind[i+1])+1)
        
        
        self.index_segment_=index_array

    def plot_data(self):

        energy_p = Preprocessing.Preprocessing.energy
        intensity_p = Preprocessing.Preprocessing.intensity
        conv_sobel = self.signal
        filtered_sig = self.signal


        fig, axes = plt.subplots(3, 1, gridspec_kw={'height_ratios': [2,3,3]}, dpi=500)
        fig.set_size_inches(12, 4)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((-2, 2)) 

        ax1, ax2, ax3 = axes[0],axes[1], axes[2]
        l1,=ax1.plot(energy_p, intensity_p, color='black')


        for ax in axes:
            ax.yaxis.set_major_formatter(formatter)
            ax.tick_params(direction='in')
            ax.set_xlim(0,max(energy_p))
            ax.spines["right"].set_visible(False)
            ax.spines['left'].set_position(('outward', 10))

        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_ylim(0,max(intensity_p))
        ax1.set_yticks([0.,25000.])
        ax1.spines["bottom"].set_visible(False)

        
        ax2.set_ylim(-4e8,4e8)
        ax3.set_ylim(-4e8,4e8)
        
        ax2.set_yticks([-4e8,4e8])
        ax3.set_yticks([-4e8,4e8])

        l2,=ax2.plot(energy_p, filtered_sig, clip_on=False, color='goldenrod')
        ax2.set_ylabel('G (a.u.)')

        ax1.spines["top"].set_visible(False)
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        


        ax3.spines['bottom'].set_position(('outward', 10))
        ax3.spines["top"].set_visible(False)


        for i in self.pos_bin:
            ax3.fill_between(energy_p[i], filtered_sig[i], color='darkred', zorder=2)
            lf1,=ax3.plot(energy_p[i], filtered_sig[i], color='darkred', zorder=2)

        for j in self.neg_bin: 
            ax3.fill_between(energy_p[j], filtered_sig[j], color='lightsteelblue', zorder=2)
            lf2,=ax3.plot(energy_p[j], filtered_sig[j], color='lightsteelblue', zorder=2)


        ax3.set_ylabel('G (a.u.)')
        ax3.set_xlabel('Energy (keV)')


        ax3.legend([l1, l2, lf1, lf2],["Original signal", "Transformed signal",
                                            "Positive distribution (feature 1)"
                                            ,"Negative distribution (feature 2)" ], 
                    loc='lower right', bbox_to_anchor=(1.05, -1.6), frameon=False)

        plt.subplots_adjust(hspace=0.4)
        plt.show()
       
