import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.signal import detrend



def raspi_import(path, channels=5):

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    # sample period is given in microseconds, so this changes units to seconds
    sample_period *= 1e-6
    return sample_period, data

def fft_data_func(sample_period, data):
    start_time = 0.01
    z_padding = True

    fs = 1/sample_period
    begin_sample = int(start_time*fs)

    Q = data[begin_sample:, 0] - np.mean(data[begin_sample:,0])
    I = data[begin_sample:, 1] - np.mean(data[begin_sample:,1])

    cmpx_signal = I + 1j*Q

    window_func = np.hanning(len(cmpx_signal))
    cmpx_signal *= window_func

    if z_padding:
        N = 2**15
    else:
        N = len(cmpx_signal)
    
    fft_data = np.fft.fftshift(np.fft.fft(cmpx_signal, n=N))
    freq = np.fft.fftshift(np.fft.fftfreq(N, d = 1/fs))

    magnitude = np.abs(fft_data)
    magnitude_db = 20 * np.log10(magnitude)

    return freq, magnitude_db, fft_data

def plot_fft_doppler(freq, magnitude, f_d = None, noise = None):

    plt.figure(figsize=(10, 6))
    plt.plot(freq, magnitude) 

    if f_d is not None:
        idx = np.argmin(np.abs(freq - f_d))
        amplitude = magnitude[idx]
        plt.plot(f_d, amplitude, 'mo', label = f"Dopplerskiftet = {f_d}") 

    if noise is not None:
        plt.axhline(y=20 * np.log10(np.sqrt(noise)), color='salmon', linestyle='--', label='Støygulv')

    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid()
    plt.legend()
    #plt.ylim(0,400)
    plt.title('Dopplerspektrum: Negativ hastighet')
    #plt.show()
    #plt.savefig('frekvens_neg_hastighet', dpi = 700)
 
def compute_snr(freqs, fft_data, fD):
    # Beregn spektral effekt
    power_spectrum = np.abs(fft_data)**2

    signal_band=(fD-50, fD+50)
    noise_band=(1000, 5000)
    # Indekser for signal- og støyområder
    signal_idx = np.logical_and(freqs >= signal_band[0], freqs <= signal_band[1])
    noise_idx = np.logical_and(freqs >= noise_band[0], freqs <= noise_band[1])

    # Beregn effekt (gjennomsnitt eller sum)
    P_signal = np.mean(power_spectrum[signal_idx])
    P_noise = np.mean(power_spectrum[noise_idx])

    snr_db = 10 * np.log10(P_signal / P_noise)

    return snr_db, P_noise

if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/SR/radar_rev5.bin')
    freq, magnitude, fft_data = fft_data_func(sample_period, data)
    snr_dB, P_noise = compute_snr(freq,fft_data, -257)
    plot_fft_doppler(freq, magnitude, -257, P_noise)
    print('SNR: ', snr_dB, 'Støygulv: ', P_noise)
