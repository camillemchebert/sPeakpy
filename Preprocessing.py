class Preprocessing:
    """ Preprocessing class """
    def __init__(self,energy, intensity, filter_type=None, ws=None):
        
        self.intensity = intensity
        self.sobel_ = self.sobel_filter()

        Preprocessing.energy = energy
        try:
            if filter_type == None:
                Preprocessing.signal_ = sobel_
            else:
                Preprocessing.signal_ = self.smooth_filter(filter_type, ws)
        except NameError:
            print('Need to be a valid data filter')
            raise
                    

    def sobel_filter(self):
        sobel_v = np.array([1,0,-1])
        conv_sobel=np.convolve(sobel_v, self.intensity, 'same')
        sobel=conv_sobel-np.mean(conv_sobel)
        return sobel
    
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
                ws = 8
            mean_vector = np.ones(ws)/ws
            movav_filter=np.convolve(mean_vector, self.sobel_, 'same')
            print('Signal transformed with moving average filter using window size of %d' %ws)
            return movav_filter
        
        else:
            raise NameError('Filter used is not contained in the library {sav_gol, fft, moving_avg}')
