import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import detrend, windows     # windows.hann is here


def raspi_import(path, channels=2):
    with open(path, 'rb') as fid:               # use 'rb' for binary
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        data = data.reshape((-1, channels))
    sample_period *= 1e-6
    return sample_period, data


if __name__ == "__main__":
    sample_period, data = raspi_import(
        sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/SR/radar_rev2.bin'
    )
    I = detrend(data[:,0])
    Q = detrend(data[:,1])
    cmpx_signal = I + 1j*Q

    # --- apply a Hann window ---
    n = len(cmpx_signal)
    win = windows.hann(n, sym=False)           # create the window
    windowed = cmpx_signal * win               # window-multiply

    # --- FFT on windowed data ---
    fft_data = np.fft.fft(windowed)
    frekvenser = np.fft.fftfreq(n, d=sample_period)

    # magnitude in dB, normalized
    magnitude = np.abs(fft_data)
    magnitude_db = 20 * np.log10(magnitude)
    magnitude_db -= np.max(magnitude_db)

    # plot only the positive half
    #half = n // 2
    half = n // 2
    # find index of the max in the positive-freq half
    idx_peak = np.argmax(magnitude_db[:half])
    x_peak   = frekvenser[idx_peak]
    y_peak   = magnitude_db[idx_peak]

    # then in your plotting:
    plt.figure(figsize=(10,6))
    plt.plot(frekvenser[:half],
            magnitude_db[:half],
            label='Windowed spectrum',
            zorder=1)

    # big purple circle on the real peak
    plt.plot(x_peak, y_peak,
            marker='o',
            markersize=8,
            linestyle='None',
            color='purple',
            label=f'Signifikant topp: {x_peak:.1f} Hz',
            zorder=5)

    # optional arrow annotation
    plt.annotate(f"{x_peak:.1f} Hz",
                xy=(x_peak, y_peak),
                xytext=(x_peak, y_peak-20),
                textcoords='data',
                arrowprops=dict(arrowstyle='->', color='purple'),
                ha='center',
                zorder=6)

    plt.axhline(y=-69.27, color='salmon', linestyle='--', label='Estimert SNR')
    """plt.figure(figsize=(10, 6))
    plt.plot(frekvenser[:half], magnitude_db[:half], label='Windowed spectrum')
    plt.plot(106.3, -46.8, 'o', color='indigo',
             label='Signifikant topp: 106.3 Hz')
    plt.axhline(y=-69.27, color='salmon', linestyle='--',
                label='Estimert SNR')
    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.xlim(0, 400)
    plt.ylim(-100, 0)
    plt.grid()
    plt.legend()
    plt.title('Frekvensspekter (med Hann-vindu)')"""
    #plt.savefig('frekvens_negativ_windowed.png', dpi=700)
    plt.show()
