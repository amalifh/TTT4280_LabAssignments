import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.signal import detrend



def raspi_import(path, channels=2):

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data

if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/SR/radar_rev4.bin')
    channel_data = data[:,0] #I-signal
    channel_data2 = data[:,1] #Q-signal

    I_signal = detrend(channel_data)
    Q_signal = detrend(channel_data2)
    window_func = len

    cmpx_signal = I_signal + 1j*Q_signal

    fft_data = np.fft.fft(cmpx_signal)

    n = len(cmpx_signal)
    n_z = 2**15
    frekvenser = np.fft.fftfreq(n, d=sample_period)

    magnitude = np.abs(fft_data)

    magnitude_db = 20 * np.log10(magnitude)

    magnitude_db_normalisert = magnitude_db - np.max(magnitude_db)

    plt.figure(figsize=(10, 6))
    plt.plot(frekvenser[:n//2], magnitude_db_normalisert[:n//2]) 
    plt.plot(93.4,-41.1, 'o', color = 'indigo', label = 'Signifikant topp: 86.2 Hz')
    plt.axhline(y = -69.27, color = 'salmon', linestyle = '--', label = 'esitmert SNR')
    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.xlim(0,400)
    plt.ylim(-100, 0)
    plt.grid()
    plt.legend()
    #plt.ylim(0,400)
    plt.title('Frekvensspekter: negativ hastighet')
    #plt.show()
    plt.savefig('frekvens_negativ', dpi = 700)
