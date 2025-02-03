import numpy as np
import sys
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq 
from scipy.signal import windows
#matplotlib inline

def raspi_import(path, channels=5):
    with open(path , 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0] 
        data = np.fromfile(fid, dtype='uint16').astype('float64') 
        data = data.reshape((-1, channels))
    sample_period *= 1e-6 
    return sample_period , data

if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'test10.bin')
    channel_data = data[:,4] #ADC 1
# FFT
    fft_data = fft(channel_data)
# Frekvensakse
    n = len(channel_data)
    frekvenser = fftfreq(n, d=sample_period)
# Amplitude
    magnitude = np.abs(fft_data)
#sx = np.square(magnitude) # i dB
    magnitude_db = 20 * np.log10(magnitude)
# Normalisert amplitude
    magnitude_db_normalisert = magnitude_db - np.max(magnitude_db)
# Plot
    plt.figure(figsize=(10, 6))
    plt.plot(frekvenser[:n//2], magnitude_db_normalisert[:n//2]) 
    plt.xlabel('Frekvens (Hz)')
    plt.ylabel('Amplitude')
    plt.xlim(0,200)
    #plt.ylim(0,400)
    plt.title('Frekvensspektrum av sinusbølge')
    #plt.show()

    plt.savefig('Frekvensspektrum_ADC5.png', dpi = 700)