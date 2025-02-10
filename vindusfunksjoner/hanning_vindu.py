import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from scipy.fft import fft, fftfreq


def raspi_import(path, channels=5):
    with open(path, 'rb') as fid: 
        sample_period = np.fromfile(fid, count=1, dtype=np.float64)[0] 
        data = np.fromfile(fid, dtype=np.uint16).astype('float64')
        data = data.reshape((-1, channels))
    sample_period *= 1e-6 
    return sample_period, data

filename = 'testBIns/test13.bin' 
sample_period, data = raspi_import(filename)

signal = data[:, 0]
signal -= np.mean(signal)  

sampling_rate = 1 / sample_period

window = windows.hann(len(signal))

window_signal = signal * window

fft_data = fft(window_signal)
n = len(signal)
frekvenser = fftfreq(n, d=sample_period)

magnitude = np.abs(fft_data)
magnitude_db = 20 * np.log10(magnitude)
magnitude_db_normalisert = magnitude_db - np.max(magnitude_db)


plt.figure(figsize=(10, 6))
plt.plot(frekvenser, magnitude_db_normalisert, label='ADC 1 med Hanning-vindu')
plt.xlabel('Frekvens (Hz)')
plt.ylabel('Amplitude (dB)')
#plt.legend()
plt.xlim(-4000, 4000) 
plt.title('Hanning-vindu')
plt.grid()
plt.savefig('Frekvensspektrum_Hanning.png', dpi=700)
#plt.show()



