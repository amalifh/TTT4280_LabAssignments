import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.signal import butter, filtfilt

# Must access the data, all 3 channels
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
    # Load the data
    r, g, b = access_data('TTT4280_LabAssignments/Lab3/test4.txt')
    t = np.arange(0, 30, 30/len(r))
    
    print("Length of signal:", len(r))
    print("Length of time array:", len(t))
    
    # Detrend the data
    r_detrend = detrend(r)
    g_detrend = detrend(g)
    b_detrend = detrend(b)

    # Define the bandpass filter function
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    # Apply the bandpass filter to the detrended data
    fs = len(r) / 30  # Sampling frequency (samples per second)
    lowcut = 0.5  # Lower bound of the bandpass filter (in Hz)
    highcut = 3.0  # Upper bound of the bandpass filter (in Hz)

    r_filtered = bandpass_filter(r_detrend, lowcut, highcut, fs)
    g_filtered = bandpass_filter(g_detrend, lowcut, highcut, fs)
    b_filtered = bandpass_filter(b_detrend, lowcut, highcut, fs)

    # -------------------------------
    # Implementing FFT on the detrended data
    # -------------------------------
    N = len(r)  # Number of samples
    # Compute FFT for each channel (using detrended data)
    r_fft = np.fft.fft(r_detrend)
    g_fft = np.fft.fft(g_detrend)
    b_fft = np.fft.fft(b_detrend)

    # Frequency bins
    freq = np.fft.fftfreq(N, d=(t[1]-t[0]))

    # Only take the positive half of the spectrum for plotting
    pos_mask = freq >= 0
    freq = freq[pos_mask]
    r_fft = r_fft[pos_mask]
    g_fft = g_fft[pos_mask]
    b_fft = b_fft[pos_mask]

    # Plot the magnitude spectrum for each channel
    plt.figure(figsize=(10, 6))
    plt.plot(freq, np.abs(r_fft), label="R")
    plt.plot(freq, np.abs(g_fft), label="G")
    plt.plot(freq, np.abs(b_fft), label="B")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.title("FFT Magnitude Spectrum")
    plt.grid()
    plt.legend()
    plt.show()
