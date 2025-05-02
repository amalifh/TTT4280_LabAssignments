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
    fs = 1 / sample_period
    start_time = 0.1
    start_sample = int(start_time * fs)

    I_signal = data[start_sample:, 0] - np.mean(data[start_sample:, 0])
    Q_signal  = data[start_sample:, 1] - np.mean(data[start_sample:, 1])


    cmpx_signal = I_signal + 1j*Q_signal

    window_func = np.hanning(len(cmpx_signal))
    cmpx_signal *= window_func

    N = 2**15

    fft_data = np.fft.fftshift(np.fft.fft(cmpx_signal, n=N))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d = 1/fs))

    magnitude = np.abs(fft_data)

    magnitude_db = 20 * np.log10(np.abs(fft_data))

    plt.figure(figsize=(10, 6))
    plt.plot(freq, magnitude_db) 
    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid()
    #plt.legend()
    #plt.ylim(0,400)
    plt.title('Frekvensspekter: negativ hastighet')
    plt.show()
    #plt.savefig('frekvens_negativ', dpi = 700)
<<<<<<< HEAD

=======
>>>>>>> d80963c32a8213cbd94547ba2d2257ceb723c13b
