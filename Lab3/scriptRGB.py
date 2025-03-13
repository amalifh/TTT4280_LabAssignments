import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend
from scipy.signal import butter, filtfilt

#Must access the data, all 3 channels
def access_data(filename):
    with open(filename, 'r') as f:
        r = []
        g = []
        b = []

        lines = f.readlines()
        for i in range(3, len(lines)):
            line = lines[i].split(' ')
            r.append(float((line[0]).strip()))
            g.append(float((line[1]).strip()))
            b.append(float((line[2]).strip()))
    return r, g, b



if __name__ == "__main__": #nice metode for filbruk osv
    r, g, b =access_data('Lab3/theosfinger5.txt')
    t = np.arange(0,30,30/len(r))
    print(len(r))
    print(len(t))
    r_detrend = detrend(r)
    g_detrend = detrend(g)
    b_detrend = detrend(b)

    # Define the bandpass filter
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    # Apply the bandpass filter to the detrended data
    fs = len(r) / 30  # Sampling frequency
    lowcut = 0.5  # Lower bound of the bandpass filter (in Hz)
    highcut = 3.0  # Upper bound of the bandpass filter (in Hz)

    r_filtered = bandpass_filter(r_detrend, lowcut, highcut, fs)
    g_filtered = bandpass_filter(g_detrend, lowcut, highcut, fs)
    b_filtered = bandpass_filter(b_detrend, lowcut, highcut, fs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(t, r, color ='red')
    ax1.plot(t, g, color = 'green')
    ax1.plot(t, b, color = 'blue')
    plt.grid()
    plt.xlabel("Tid [s]")

    ax2.plot(t, r_detrend, color ='red')
    ax2.plot(t, g_detrend, color = 'green')
    ax2.plot(t, b_detrend, color = 'blue')
    plt.grid()
    plt.xlabel("Tid [s]")

    ax3.plot(t, r_filtered, color ='red')
    ax3.plot(t, g_filtered, color = 'green')
    ax3.plot(t, b_filtered, color = 'blue')
    plt.grid()
    plt.xlabel("Tid [s]")

    plt.show()
    #plt.savefig("pulsmåling_ingenprosess")