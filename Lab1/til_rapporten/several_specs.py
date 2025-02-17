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
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'testBIns/test13.bin')
    channel_data = data[:,0] #ADC 1
    channel_data2 = data[:,1]
    channel_data3 = data[:,2]
    channel_data4 = data[:,3]
    channel_data5 = data[:,4]

    fft_data = fft(channel_data)
    fft_data2 = fft(channel_data2)
    fft_data3 = fft(channel_data3)
    fft_data4 = fft(channel_data4)
    fft_data5 = fft(channel_data5)

    n = len(channel_data)
    n2 = len(channel_data2)
    n3 = len(channel_data3)
    n4 = len(channel_data4)
    n5 = len(channel_data5)

    frekvenser = fftfreq(n, d=sample_period)
    frekvenser2 = fftfreq(n2, d=sample_period)
    frekvenser3 = fftfreq(n3, d=sample_period)
    frekvenser4 = fftfreq(n4, d=sample_period)
    frekvenser5 = fftfreq(n5, d=sample_period)

    magnitude = np.abs(fft_data)
    magnitude2 = np.abs(fft_data2)
    magnitude3 = np.abs(fft_data3)
    magnitude4 = np.abs(fft_data4)
    magnitude5 = np.abs(fft_data5)

    magnitude_db = 20 * np.log10(magnitude)
    magnitude_db2 = 20 * np.log10(magnitude2)
    magnitude_db3 = 20 * np.log10(magnitude3)
    magnitude_db4 = 20 * np.log10(magnitude4)
    magnitude_db5 = 20 * np.log10(magnitude5)

    magnitude_db_normalisert = magnitude_db - np.max(magnitude_db)
    magnitude_db_normalisert2 = magnitude_db2 - np.max(magnitude_db2)
    magnitude_db_normalisert3 = magnitude_db3 - np.max(magnitude_db3)
    magnitude_db_normalisert4 = magnitude_db4 - np.max(magnitude_db4)
    magnitude_db_normalisert5 = magnitude_db5 - np.max(magnitude_db5)

    plt.figure(figsize=(10, 6))
    plt.plot(frekvenser[:n//2], magnitude_db_normalisert[:n//2], label = 'ADC 1') 
    plt.plot(frekvenser2[:n//2], magnitude_db_normalisert2[:n//2], label = 'ADC 2')
    plt.plot(frekvenser3[:n//2], magnitude_db_normalisert3[:n//2], label = 'ADC 3')
    plt.plot(frekvenser4[:n//2], magnitude_db_normalisert4[:n//2], label = 'ADC 4')
    plt.plot(frekvenser5[:n//2], magnitude_db_normalisert5[:n//2], label = 'ADC 5')

    plt.xlabel('Frekvens (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.xlim(0,200)
    plt.title('Frekvensspektrum av sinusbølge')
    plt.show()