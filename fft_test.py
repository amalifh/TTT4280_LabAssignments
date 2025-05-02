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

def compute_fft(sample_period, data, start_time=0.1, zero_pad=True):
    fs = 1 / sample_period
    start_sample = int(start_time * fs)

    Q = data[start_sample:, 0] - np.mean(data[start_sample:, 0]) # fjerner DC-komponent
    I = data[start_sample:, 1] - np.mean(data[start_sample:, 1])

    complex_signal = I + 1j * Q  # kombinerer I og Q til et komplekst signal

    # Bruk Hann-vindu for å redusere spektrallekkasje
    window = np.hanning(len(complex_signal))
    complex_signal *= window

    # Zero-padding for høyere frekvensoppløsning 
    N = 2**14 if zero_pad else len(complex_signal)

    fft_data = np.fft.fftshift(np.fft.fft(complex_signal, n=N))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))
    magnitude = 20 * np.log10(np.abs(fft_data))

    return freqs, magnitude, fft_data

sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/SR/radar_rev5.bin')

freq, magnitude, fft_data = compute_fft(sample_period, data)


def plot_doppler_spectrum(freqs, magnitude, f_doppler=None):
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude, label="Spektrum")
    
    if f_doppler is not None:
        # Finn amplitudeverdien ved Dopplerpeaken
        idx = np.argmin(np.abs(freqs - f_doppler))
        amp = magnitude[idx]

        plt.axvline(f_doppler, color='red', linestyle='--', label=f'Dopplerpeak: {f_doppler:.1f} Hz')
        plt.plot(f_doppler, amp, 'ro')  # Rød prikk

    plt.title("Dopplerspektrum (kompleks FFT)")
    plt.xlabel("Frekvens [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.xlim(-400, 400)
    plt.ylim(40, 140)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_doppler_spectrum(freq, magnitude)

