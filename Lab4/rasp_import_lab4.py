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


# Import data from bin file
if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/SS/radar_slow1.bin')
    
    delta = 0.8e-3
    channel_data = data[:,0]
    channel_data *= delta
    channel_data = detrend(channel_data,axis=-1,type='linear',bp=0,overwrite_data=False)

    channel_data_2 = data[:,1]
    channel_data_2 *= delta
    channel_data_2 = detrend(channel_data_2,axis=-1,type='linear',bp=0,overwrite_data=False)

    t = np.arange(0,len(channel_data))

    I_signal_cos = np.cos(channel_data)
    Q_signal_sin = np.sin(channel_data_2)

    cmplx = I_signal_cos + 1j*Q_signal_sin

    plt.plot(np.real(cmplx), np.imag(cmplx), label = 'I-Q plott')

  

    plt.title('Rådata fra radar: Lav hastighet')
    plt.xlabel('Tid[s]')
    #plt.xlim(0,0.05)
    #plt.ylim(-1,1)
    plt.ylabel('Spenning [V]')
    plt.grid()

    plt.legend(loc = 'upper right')


    plt.show()
    

    #plt.savefig('lab4_slow1_rådata_zoom', dpi = 700)



