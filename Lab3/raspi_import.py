import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend
import numpy as np
import sys
import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq 
from scipy.signal import windows


def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.

    Returns sample period and a (`samples`, `channels`) `float64` array of
    sampled data from all `channels` channels.

    Example (requires a recording named `foo.bin`):
    ```
    >>> from raspi_import import raspi_import
    >>> sample_period, data = raspi_import('foo.bin')
    >>> print(data.shape)
    (31250, 5)
    >>> print(sample_period)
    3.2e-05

    ```
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data


# Import data from bin file
if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/Speed-Slow/radar_slow1.bin')
    
    delta = 0.8e-3
    channel_data = data[:,0]
    channel_data *= delta
    channel_data = detrend(channel_data,axis=-1,type='linear',bp=0,overwrite_data=False)

    channel_data_2 = data[:,1]
    channel_data_2 *= delta
    channel_data_2 = detrend(channel_data_2,axis=-1,type='linear',bp=0,overwrite_data=False)

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
    plt.show()




    t = np.arange(0,1,1/31250)

    plt.plot(t, channel_data, label = 'ADC 1')
    plt.plot(t, channel_data_2, label = 'ADC 2')
  


    #plt.xlim(0,0.1)

    plt.title('Plot av rådata')
    plt.xlabel('Tid [s]')
    plt.xlim(0,0.05)
    plt.ylim(-1,1)
    plt.ylabel('Amplitude [V]')
    plt.grid()

    plt.legend(loc = 'upper right')


    plt.show()

    #plt.savefig('lab2_klapp10_180grader.png', dpi = 700)
