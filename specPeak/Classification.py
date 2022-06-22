import specPeak.Preprocessing as Preprocessing
import specPeak.Segmentation as Segmentation
#import specPeak.Data as data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
#from scipy.stats import skewnorm

class Classification:
    """ Classification class """

    def __init__(self, index_segment_, threshold):
        
        if threshold==None:
            self.threshold = 0.0
        else:
            self.threshold = threshold
            
        self.energy = Preprocessing.Preprocessing.energy
        self.intensity = Preprocessing.Preprocessing.intensity
        self.signal_ = Preprocessing.Preprocessing.signal_
        self.index_segment_=index_segment_
        
        self.index_segment_bin = []
        self.index_temp_bin = []
        
        self.qualification_=self.qualification()
        self.quantification_=self.quantification()
        
    def qualification(self):
        self.qual_score = []
        

        for ind in self.index_segment_:
                
                sobel_bin=self.signal_[ind]
                
                if (max((sobel_bin))-min((sobel_bin))) !=0:
                    y = (sobel_bin)
                    y = y/(max((y))-min((y)))
                    x = np.linspace(-1, 1, len(y))
                    
                W = np.zeros([len(x), len(x)])
                D = np.zeros([len(x), len(x)])
                I = np.zeros([len(x), len(x)])
                L = np.zeros([len(x), len(x)])

                for i in np.arange(len(x)):
                    for j in np.arange(len(x)):
                        W[i,j] = np.exp(-1*abs(((y[i]**2+x[i]**2)**.5)-((y[j]**2+x[j]**2)**.5))**2)/(2*np.std((y))**2)

                for i in np.arange(len(x)):
                    D[i,i] = sum(W[i][:])

                for i in np.arange(len(x)):
                    I[i,i]=1

                L =(D**0.5)*((D**0.5)-W)*(D**0.5)

                e, v = np.linalg.eig(L)

                part_vector = v[np.argmin(e)][:]

                ind_ =np.where(part_vector==1)[0][0]

                score = 1-(abs(len(part_vector[ind_:])-len(part_vector[:ind_]))/len(part_vector))
                
                if score>self.threshold:
                    self.index_temp_bin.append(ind)
                    self.qual_score.append(score)
                    #ax[0].axvline(x=x[part_vector==1], color='lightgray', alpha =0.3)
##                    ax[0].plot(x, y, color='gray', alpha=.8)
##                    ax[0].set_ylim(-1, 1)
##                    ax[0].legend({'Peak segments'}, frameon=False)
                    
                else:
                    Segmentation.Segmentation.index_noise_bin.append(ind)
        
        for ind in Segmentation.Segmentation.index_noise_bin:
            energy_bin=np.array(self.energy[ind])
            sobel_bin=self.signal_[ind]
            if (max((sobel_bin))-min((sobel_bin))) !=0:
                y = (sobel_bin)/(max((sobel_bin))-min((sobel_bin)))
                x = np.linspace(-1, 1,len(y) )
                
##                ax[1].plot(x, y, color='lavender', alpha=1)
##        ax[1].set_ylim(-1, 1)
##        ax[1].set_ylabel('Normalized G(x)')
##        ax[1].set_xlabel('Normalized x')
##        ax[0].set_ylabel('Normalized G(x)')
##        
##        ax[1].legend({'Noise segments'}, frameon=False)
        plt.figure()
        plt.hist(self.qual_score)
        plt.show()

        
                
        
    def quantification(self):
        self.percent = []
        
        noise_sum_val=[]
        peak_sum_vall=[]
        peak_bin=[]
        all_bin=[]
        self.threshold = np.mean(self.qual_score)

        for j in Segmentation.Segmentation.index_noise_bin:
            if not np.sum(abs(self.signal_[j]))==0:
                noise_sum_val.append((np.sum(abs(self.signal_[j])))*len(self.signal_[j]))
                
        for j in self.index_temp_bin:
            if not np.sum(abs(self.signal_[j]))==0:
                peak_sum_vall.append((np.sum(abs(self.signal_[j])))*len(self.signal_[j]))

        
        peak_sum_vall= np.log10(peak_sum_vall)
        noise_sum_val = np.log10(noise_sum_val)
        
        peak_sum_val = peak_sum_vall
        
        noise_sum_val=np.sort(noise_sum_val)
        peak_sum_val=np.sort(peak_sum_val)



        all_bin = np.concatenate((noise_sum_val, peak_sum_val))
        all_bin=np.sort(all_bin)
        dist_all = norm(np.mean(all_bin), np.std(all_bin))
        
        noise_weight = len(noise_sum_val)/len(all_bin)
        peak_weight = len(peak_sum_val)/len(all_bin)

        self.entropy = (peak_weight*len(all_bin)*np.log(1/peak_weight)+noise_weight*len(all_bin)*np.log(1/noise_weight))/len(all_bin)

        probabilities_noise= (1/(np.std(noise_sum_val)*np.sqrt(2*np.pi)))*np.exp(-0.5*(((all_bin)-np.mean(noise_sum_val))/np.std(noise_sum_val))**2)
        probabilities_peak= (1/(np.std(peak_sum_val)*np.sqrt(2*np.pi)))*np.exp(-0.5*(((all_bin)-np.mean(peak_sum_val))/np.std(peak_sum_val))**2)


        mm = np.zeros([len(probabilities_noise), len(probabilities_peak)])

        for i in np.arange(len(probabilities_noise)):
            for j in np.arange(len(probabilities_peak)):
                mm[i]=probabilities_noise[i]
                mm[j]=probabilities_peak[j]

        probabilities_all = dist_all.pdf(all_bin)
        probabilities_noise=probabilities_noise/(sum(probabilities_noise)+sum(probabilities_peak))
        probabilities_peak=probabilities_peak/(sum(probabilities_peak)+sum(probabilities_peak))

        ent = ((probabilities_noise)*(np.log(probabilities_noise))/(np.log(probabilities_peak)))-(probabilities_noise*np.log(probabilities_noise))

        cum_sum = []
        sum_val=0
        for i in np.arange(len(probabilities_peak)):
            sum_val += (probabilities_peak[i]*np.log(1/probabilities_peak[i]))#/probabilities_noise[i]))
            cum_sum.append(sum_val)
            
        cum_sum2 = []
        sum_val2=0
        for i in np.arange(len(probabilities_peak)):
            sum_val2 += (probabilities_noise[i]*np.log(1/probabilities_noise[i]))#/probabilities_peak[i]))
            cum_sum2.append(sum_val2)

        cross_entropy = np.array(cum_sum)-np.array(cum_sum2)

        qind=0
        for i in self.index_temp_bin:
            if not np.mean(abs(self.signal_[i]))-np.mean(self.signal_[i])==0:
                peak_criterion = (np.log10(((np.sum(abs(self.signal_[i]))))*len(self.signal_[i])))*self.qual_score[qind]
                qind+=1
                if peak_criterion>(all_bin[np.argmin(cross_entropy)]*np.mean(self.qual_score)):
                    threshold_criterion=(((peak_criterion))/max(peak_sum_val))
                    self.index_segment_bin.append(i)
                    peak_bin.append(peak_criterion)
                    self.percent.append(threshold_criterion)

                else:
                    Segmentation.Segmentation.index_noise_bin.append(i)


        

        plt.figure()
        plt.plot(all_bin, cross_entropy)
        plt.axvline(x=all_bin[np.argmin(cross_entropy)])

        plt.figure()
        plt.hist(peak_sum_val, alpha=0.4)
        plt.hist(noise_sum_val, alpha=0.4)
        plt.axvline(x=self.entropy*np.mean(self.qual_score), linestyle='--')
        plt.axvline(x=all_bin[np.argmax(cross_entropy)])
           

        

            
