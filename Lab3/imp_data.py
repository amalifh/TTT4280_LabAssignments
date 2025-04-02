import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.signal import butter, filtfilt

def access_data(filename):
    with open(filename, 'r') as f:
        r = []
        g = []
        b = []
        lines = f.readlines()
        for i in range(3, len(lines)):
            line = lines[i].split(' ')
            r.append(float(line[0].strip()))
            g.append(float(line[1].strip()))
            b.append(float(line[2].strip()))
    return r, g, b

if __name__ == "__main__":
   
    r, g, b = access_data('Lab3/reflektans_1.txt')
    t = np.arange(0, 30, 30/len(r))
    
    print("Length of signal:", len(r))
    print("Length of time array:", len(t))
    

    r_detrend = detrend(r)
    g_detrend = detrend(g)
    b_detrend = detrend(b)


    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    fs = len(r) / 30  
    lowcut = 0.5 
    highcut = 3.0 

    r_filtered = bandpass_filter(r_detrend, lowcut, highcut, fs)
    g_filtered = bandpass_filter(g_detrend, lowcut, highcut, fs)
    b_filtered = bandpass_filter(b_detrend, lowcut, highcut, fs)


    N = len(r)  # Number of samples

    r_fft = np.fft.fft(r_detrend)
    g_fft = np.fft.fft(g_detrend)
    b_fft = np.fft.fft(b_detrend)

    freq = np.fft.fftfreq(N, d=(t[1]-t[0]))

 
    pos_mask = freq >= 0
    freq = freq[pos_mask]
    r_fft = r_fft[pos_mask]
    g_fft = g_fft[pos_mask]
    b_fft = b_fft[pos_mask]

    plt.figure(figsize=(10, 6))
    plt.plot(freq, np.abs(r_fft), label="R", color = 'red')
    plt.plot(freq, np.abs(g_fft), label="G", color = 'green')
    plt.plot(freq, np.abs(b_fft), label="B", color = 'blue')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("FFT Magnitude Spectrum")
    plt.grid()
    plt.legend()
    plt.show()


hr_min = 0.5 
hr_max = 3.0 


band_mask = (freq >= hr_min) & (freq <= hr_max)


g_fft_magnitude = np.abs(g_fft)[band_mask]
band_freqs = freq[band_mask]


peak_index = np.argmax(g_fft_magnitude)

pulse_frequency = band_freqs[peak_index]

pulse_rate_bpm = pulse_frequency * 60

print("Estimated Pulse Rate (BPM) from green channel:", pulse_rate_bpm)


delta = 0.1
signal_indices = np.where(np.abs(freq - pulse_frequency) < delta)[0]


signal_power = np.sum(np.abs(r_fft[signal_indices])**2)

noise_mask = ((freq >= hr_min) & (freq <= hr_max)) & (np.abs(freq - pulse_frequency) >= delta)
noise_power = np.sum(np.abs(r_fft[noise_mask])**2)

if noise_power == 0:
    print("Noise power is zero; cannot compute SNR.")
else:
    snr_db = 10 * np.log10(signal_power / noise_power)
    print("Estimated SNR (dB) for red channel pulse:", snr_db)