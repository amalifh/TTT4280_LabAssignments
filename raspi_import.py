import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend


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
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab2/klapp_test/klapp12.bin')
    
    delta = 0.8e-3
    channel_data = data[:,0]
    channel_data *= delta
    channel_data = detrend(channel_data,axis=-1,type='linear',bp=0,overwrite_data=False)

    channel_data_2 = data[:,1]
    channel_data_2 *= delta
    channel_data_2 = detrend(channel_data_2,axis=-1,type='linear',bp=0,overwrite_data=False)

    channel_data_3 = data[:,2]
    channel_data_3 *= delta
    channel_data_3 = detrend(channel_data_3,axis=-1,type='linear',bp=0,overwrite_data=False)

    channel_data_4 = data[:,3]
    channel_data_4 *= delta
    channel_data_4 = detrend(channel_data_4,axis=-1,type='linear',bp=0,overwrite_data=False)

    channel_data_5 = data[:,4]
    channel_data_5 *= delta
    channel_data_5 = detrend(channel_data_5,axis=-1,type='linear',bp=0,overwrite_data=False)



    t = np.arange(0,1,1/31250)

    plt.plot(t, channel_data, label = 'ADC 1')
    plt.plot(t, channel_data_2, label = 'ADC 2')
    #plt.plot(t, channel_data_3, label = 'ADC 3')
    plt.plot(t, channel_data_4, label = 'ADC 4')
    #plt.plot(t, channel_data_5, label = 'ADC 5')


    #plt.xlim(0,0.1)

    plt.title('Opptak av klapp 180 grader på x-aksen')
    plt.xlabel('Tid [s]')
    plt.xlim(0.2,0.8)
    plt.ylabel('Amplitude [V]')
    plt.grid()

    plt.legend(loc = 'upper right')


    plt.show()

    #plt.savefig('lab2_klapp10_180grader.png', dpi = 700)




def find_delay(signal1, signal2, fs):

    # Beregn krysskorrelasjon
    correlation = np.correlate(signal1, signal2, mode='full')

    # Finn forsinkelsen ved å finne indeksen til maksimumet av den absolutte krysskorrelasjonen
    delay = np.argmax(np.abs(correlation)) - (len(signal1) - 1)
    
    # Konverter fra sample-forsinkelse til tid i sekunder
    delay_time = delay / fs

    
    return delay_time


print(find_delay(channel_data, channel_data_2, 31250))



def plot_correlation(signal1, signal2, fs):
    # Beregn krysskorrelasjonen med "full" mode
    correlation = np.correlate(signal1, signal2, mode='full')
    
    # Lag en array med tidsforsinkelser (lags)
    
    lags = np.arange(-len(signal1) + 1, len(signal1))
    # Plot krysskorrelasjonen
    plt.figure()
    plt.plot(lags, correlation)
    plt.xlabel('Forsinkelse (s)')
    plt.ylabel('Krysskorrelasjon')
    plt.title('Krysskorrelasjon mellom signal 1 og signal 2, 180 grader')
    plt.grid(True)
    plt.show()
    #plt.savefig('krysskorrelasjon_klapp10.png', dpi=700)

plot_correlation(channel_data, channel_data_2, 31250)

