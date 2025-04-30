import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
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
'''
if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/SS/radar_slow1.bin')
    
    delta = 0.8e-3
    channel_data = data[:,0]
    channel_data *= delta
    channel_data = detrend(channel_data,axis=-1,type='linear',bp=0,overwrite_data=False)

    channel_data_2 = data[:,1]
    channel_data_2 *= delta
    channel_data_2 = detrend(channel_data_2,axis=-1,type='linear',bp=0,overwrite_data=False)

    sample_freq = 1/sample_period

    N_padding = 1024

    fft_signalI = np.fft.fft(channel_data, N_padding)
    freqI = np.fft.fftfreq(N_padding, 1/sample_freq)
    A_normI = abs(fft_signalI)

    pos_freqI = freqI[:N_padding//2]
    pos_ampI = A_normI[:N_padding//2]


    fft_signalQ = np.fft.fft(channel_data_2, N_padding)
    freqQ = np.fft.fftfreq(N_padding, 1/sample_freq)
    A_normQ = abs(fft_signalQ)
    pos_freqQ = freqQ[:N_padding//2]
    pos_ampQ = A_normQ[:N_padding//2]

    plt.plot(freqI, fft_signalI)
    plt.plot(freqQ, fft_signalQ)
    #plt.plot(pos_freqI, pos_ampI)
    #plt.plot(pos_freqQ, pos_ampQ)
    plt.show()
    '''
if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/SS/radar_slow2.bin')
    channel_data = data[:,0] #I-signal
    channel_data2 = data[:,1] #Q-signal

    I_signal = detrend(channel_data)
    Q_signal = detrend(channel_data2)

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
    plt.plot(119.7,-35.7,'ro', label = 'Signifikant topp: 120 Hz')
    plt.axhline(y = -69.29, color = 'salmon', linestyle = '--', label = 'esitmert SNR')
    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.xlim(0,400)
    plt.ylim(-100, 0)
    plt.grid()
    plt.legend()
    #plt.ylim(0,400)
    plt.title('Frekvensspekter: høy hastighet')
    #plt.show()
    plt.savefig('frekvens_hoyhast1', dpi = 700)
