import numpy as np


class Preprocessing:
    """ Preprocessing class """
    def __init__(self,energy, intensity, p_res=None, b_res=None):
        
        Preprocessing.energy = energy
        Preprocessing.intensity = intensity
        self.sobel_ = self.sobel_filter()


        
        try:
            if p_res == None or b_res == None:
                Preprocessing.signal_ = self.smooth_filter('moving_avg', None)
            else:
                Preprocessing.signal_ = self.smooth_filter('moving_avg', int(p_res/b_res))
        except NameError:
            print('Need to be a valid data filter')
            raise
                    

    def sobel_filter(self):
        signal = np.zeros(len(Preprocessing.intensity))
        sobel_kernel_x = [[-1,0,1],[-2, 0, 2],[-1, 0, 1]]
        sobel_kernel_y = [[1,2,1],[0, 0, 0],[-1, -2, -1]]

        
        for i in np.arange(0,len(Preprocessing.intensity)-2):
            sum_sobel=[]
            for j in np.arange(len(sobel_kernel_x)):
                s_sobel_x=0
                s_sobel_y=0
                for k in np.arange(len(sobel_kernel_x)):
                    s_sobel_x+=(Preprocessing.intensity[i+k]**2*sobel_kernel_x[j][k])
                    s_sobel_y+=(Preprocessing.intensity[i+k]**2*sobel_kernel_y[j][k]) # no variation for Gy (equivalent to 0)
                sum_sobel.append(sum([s_sobel_x,s_sobel_y]))
            signal[i+1]=sum(sum_sobel)#(Preprocessing.intensity[i]*sobel_kernel_x[j][j])#+(Preprocessing.intensity[i+1])*0+(Preprocessing.intensity[i+2])*1+Preprocessing.intensity[i]*-2+(Preprocessing.intensity[i+1])*0+(Preprocessing.intensity[i+2])*2+Preprocessing.intensity[i]*-1+(Preprocessing.intensity[i+1])*0+(Preprocessing.intensity[i+2])*1)
            
        return signal
    
    def smooth_filter(self, filter_type, ws):
        if filter_type == 'sav_gol':
            de = 0.02
            from scipy.signal import savgol_filter
            if ws == None:
                ws = int(1/(de*3)) if int(1/(de*3))%2!=0 else int(1/(de*3))-1

            sv_sobel = savgol_filter(self.sobel_,ws,3)
            
            print('Signal transformed with Sav Gol filter with window size %d' %ws)
            return sv_sobel
    
        elif filter_type == 'fft':
            from scipy.fft import fft, fftfreq, ifft
            
            N = len(self.sobel_)
            T = 1/N
            yf = fft(self.sobel_)
            xf = fftfreq(N,T)
           # xf = fftfreq(self.sobel_.size, d=de)
            peak_freq = np.std(abs(xf))
            
            high_freq_fft = yf.copy()
            high_freq_fft[abs(xf)>peak_freq] = 0
            fft_filter = ifft(high_freq_fft)
            print('Signal transformed with Fourrier transform filter')
            return np.real(fft_filter)
        
        elif filter_type == 'moving_avg':
            if ws == None:
                ws = 7
                print('No information inputed concerning peak resolution of the instrument. Default value generated')
            mean_vector = (np.ones(ws)/ws)
            movav_filter=np.convolve(mean_vector, self.sobel_, 'same')
            print('Signal transformed with moving average filter using window size of %d' %ws)
            return movav_filter
        
        else:
            raise NameError('Filter used is not contained in the library {sav_gol, fft, moving_avg}')
