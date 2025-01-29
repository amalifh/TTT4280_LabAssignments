import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np


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
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'test7.bin')
    
    delta = 0.8e-3
    channel_data = data[:,0]
    channel_data *= delta

    t = np.arange(0,1,1/31250)

    plt.plot(t, channel_data)
    plt.xlim(0,0.2)

    plt.title('Sinusbølge med frekvens 100Hz')
    plt.xlabel('Tid [s]')
    plt.ylabel('Amplitude [V]')

    #plt.show()

    plt.savefig('sample7.png', dpi = 700)

