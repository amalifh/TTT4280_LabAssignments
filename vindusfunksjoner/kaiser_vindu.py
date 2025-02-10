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

# Kaiser-vindu med beta = 8.6
beta = 8.6
window = windows.kaiser(len(signal), beta)

window_signal = signal * window

fft_data = fft(window_signal)
n = len(signal)
frekvenser = fftfreq(n, d=sample_period)

magnitude = np.abs(fft_data)
magnitude_db = 20 * np.log10(magnitude)
magnitude_db_normalisert = magnitude_db - np.max(magnitude_db)

plt.figure(figsize=(10, 6))
plt.plot(frekvenser, magnitude_db_normalisert, label='Kaiser-vindu')
plt.xlabel('Frekvens (Hz)')
plt.ylabel('Amplitude (dB)')
#plt.legend()
plt.xlim(-2000, 2000)
plt.title('Frekvensspektrum med Kaiser-vindu')
plt.grid()
plt.savefig('Frekvensspektrum_Kaiser.png', dpi=700)
plt.show()
