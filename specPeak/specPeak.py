import numpy as np
from specPeak.Preprocessing import Preprocessing
from specPeak.Segmentation import Segmentation
from specPeak.Classification import Classification
from specPeak.Identification import Identification
import matplotlib.pyplot as plt
import os

class Peak:
    def __init__(self, 
                 energy, intensity, p_res=None, 
                 b_res=None, threshold=None, element_ID = True, dataType='EDXRF',
                 output_data = None, plot_data=True):
        """
        Identifies peak features of a spectrum with a given
            theshold that can be specified by the user

        Returns a dataframe containing the information extracted
            from the algorithm.
            [Element, Energy, Intensity, Percentage]
        
        ----------
        energy : ndarray
            Energy values of the spectrum in keV units.

        intensity : ndarray
            Intensity values of the spectrum in count units.
        
        p_res : float64 , optional
            Peak resolution of the instrument. This unit will be used
            to smooth the input spectrum with a moving average filter.
            The window size will be automatically computed with the
            inputed bin resolution (b_res).
            Default value is 140 eV.

        b_res : float64 , optional
            Bin resolution of the spectrum. This unit will be used to
            smooth the input spectrum with a moving average filter.
            The window size will be automatically computed with the
            inputed peak resolution of the instrument(p_res).
            Default value is 20 eV.
        
        threshold : float64 , optional
            Threshold percentage to use in extracting pattern information.
            The value is used for both the qualitative and quantitative
            threshold required for feature extraction. Valid entry from
            0 to 100. Recommended values between 40 and 70.
            Default value is 60.
            
        element_id : bool, optional
            Elemental identification of the spectrum. Given the selected
            peaks, the algorithm provides a dataframe with information
            regarding the qualitative approximation of the elemental
            composition of the sample.
            If set to True, dataType is required.

        dataType: string, optional
            Type of data associated with the spectrum. The information
            is used to reference the associated peaks with an element.
            Only the energy dispersive X-ray fluorescence (EDXRF) is
            currently available.
            Default is 'EDXRF'.
            
        output_data : string, optional
            If output_data is None (default), return dataframe is
            only printed in the prompt. If output_data is specified
            with a string, it provides a .csv file containing the
            information of the return dataframe.

        plot_data : bool, optional
            Plot of the spectrum with the identified peak cross
            section obtained from the algorithm.
            Default value is True.
            
        Returns
        -------

        Algorithm returns a dataframe that can be accessed through .txt/.csv
        files. For unspecified output_data name, no file is generated. The results of
        the algorithm can be accessed through get_data(). See example.

        Dataframe columns
        
        Element : Element identified through peak selection.
        Energy : Energy levels are seperated into 4 categories.
            [Ki, Kii, Li, Lii]
        Intensity : Intensity values associated to energy levels.
            Data is seperated into 4 categories.
            [Ki, Kii, Li, Lii]
        Percentage : Index indicating the level of fit to the feature
            extraction procedure. Low percent indicates poor correlation
            with a peak feature. High percent indicates adequate peak
            resolution.

        Notes
        -----
        The algorithm is seperated into 4 procedures: Preprocessing, segmentation,
        classification, and identification. These procedures have default values
        associated to required parameters. This is done to facilitate the
        functionality of the algorithm.

        Preprocessing is divided into 2 components: Signal transformation and
        data filtering. The signal is first transformed into a smoothed first
        derivative with a Sobel kernel to facilitate segmentation of the signal.
        Smoothing filter is then used to reduced instrument noise. It uses a
        moving average filter with a window size corresponding to the specification
        of the instrument. Default values are provided with a standard portable
        XRF instrument with peak resolution of 140 eV and bin resolution of 20 eV.
        (corresponding to window size of 7).

        Segmentation uses the transformed signal to identify oscillation patterns.
        The indices of the data distribution from positive and negative (and
        negative to positive) values are extracted. Indices of N-segments are
        returned for feature extraction and classification.

        Classification uses spectral clustering to identify a continuous distribution
        of positive to negative features of the transformed data. The score obtained
        by the normalized mutual info score from sklearn Python library is compared to
        the threshold value. If passed, segment is assigned as peak. If failed,
        segment is assigned as noise. This procedure corresponds to the qualitative
        classification. Then, a quantitative classification is used by comparing
        the segments classified as noise and the segments classified as peaks.
        A proportionality is extracted between the sum of the cross section of
        the peak segment and the average of the sum of all noise cross sections.
        This proportion ratio must be greated than threshold.

        The resulting peaks are identified through a reference database provided by
        NIST (National Institute of Science and Technology). The data is incorporated
        into a dataframe that can be accessed through a specified file name or
        directly from the Python prompt.

        Please note that specPeak algorithm does not quantify elemental composition.
        It only extracts peak-like features through spectral clustering and provides
        an ESTIMATE of possible elements contained in a spectrum. Percentage score
        should be regarded as a guideline and is always relative to the noise
        distribution of the spectrum. High noise levels reduces the efficiency of the
        algorithm leading to possible misidentification of peak-like features to
        elements.
        
        Example
        --------
        >>> import specPeak as sp
        >>> energy = array([0.000000e+00, 2.005000e-02, 4.010000e-02, ...,
            2.395975e+01, 2.397980e+01, 2.399985e+01], dtype=np.float64)
        >>> intensity = array([949, 774, 599, ..., 602, 674, 632], dtype=int64)
        >>> element_peaks = sp.Peak(energy, intensity)
        >>> element_peaks.get_data()
        
        """

        self.energy = energy
        self.intensity = intensity
        
        self.r=Preprocessing(self.energy, self.intensity, p_res, b_res)
        self.s=Segmentation(self.r.signal_)
        self.c=Classification(self.s.index_segment_, threshold)
        self.e=Identification(self.c.index_segment_bin, dataType, self.c.percent)

        if plot_data==True:
            self.plot_()
            
        self.get_data(output_data)

    def get_data(self, fileName=None):
        if fileName == None:
            return self.e.EDXRF_ID()
        else:
            if not os.path.isdir('Results'):
                os.mkdir('Results')
            if not os.path.isdir(os.path.join('Results', fileName+'.csv')):
                os.mkdir(os.path.join('Results',fileName+'.csv'))

    def plot_(self):
        fig, ax = plt.subplots(2,1)
        ax[0].plot(self.energy, self.intensity, color='lightgray', alpha=0.8)
        ax[1].plot(self.energy, self.intensity, color='lightgray', alpha=0.8)
        ax[1].set_yscale('log')
        for i in self.c.index_segment_bin:
            ax[0].plot(self.energy[i], self.intensity[i])
            ax[1].plot(self.energy[i], self.intensity[i])
        plt.xlabel('Energy (keV)')
        plt.ylabel('Intensity (counts)')
        plt.show()

