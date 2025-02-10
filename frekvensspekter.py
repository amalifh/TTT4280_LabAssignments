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
    channel_data = data[:,0] #ADC 1
    channel_data2 = data[:,1]
    channel_data3 = data[:,2]
    channel_data4 = data[:,3]
    channel_data5 = data[:,4]


    fft_data = fft(channel_data)

    n = len(channel_data)
    frekvenser = fftfreq(n, d=sample_period)

    magnitude = np.abs(fft_data)

    magnitude_db = 20 * np.log10(magnitude)

    magnitude_db_normalisert = magnitude_db - np.max(magnitude_db)

    plt.figure(figsize=(10, 6))
    plt.plot(frekvenser[:n//2], magnitude_db_normalisert[:n//2], label = 'ADC 1') 
    plt.xlabel('Frekvens (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.xlim(0,200)
    plt.grid()
    #plt.ylim(0,400)
    plt.title('Frekvensspektrum av sinusbølge')
    #plt.show()

    plt.savefig('Frekvensspektrum_ADC5.png', dpi = 700)