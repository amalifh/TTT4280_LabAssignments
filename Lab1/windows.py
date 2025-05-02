import numpy as np
import sys
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq 
from scipy.signal import windows

def raspi_import(path, channels=5):
    with open(path , 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0] 
        data = np.fromfile(fid, dtype='uint16').astype('float64') 
        data = data.reshape((-1, channels))
    sample_period *= 1e-6 
    return sample_period , data

if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'testBIns/test13.bin')
    channel_data = data[:,0]  # ADC 1

    # Define zero-padded length (next power of two)
    original_n = len(channel_data)
    padded_n = 2**int(np.ceil(np.log2(original_n)))  # Next power of 2

    # Apply zero-padding
    channel_data_padded = np.pad(channel_data, (0, padded_n - original_n), mode='constant')

    # FFT with zero-padding
    fft_data = fft(channel_data_padded)

    # Frequency axis
    frekvenser = fftfreq(padded_n, d=sample_period)

    # Amplitude spectrum
    magnitude = np.abs(fft_data)
    magnitude_db = 20 * np.log10(magnitude)
    magnitude_db_normalisert = magnitude_db - np.max(magnitude_db)

    
    
    # Save figure
    #plt.savefig('Frekvensspektrum_ADC5_padded.png', dpi=700)
