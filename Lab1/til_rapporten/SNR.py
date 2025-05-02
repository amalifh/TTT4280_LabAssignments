import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import windows

def raspi_import(path, channels=5):
    with open(path, 'rb') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=np.float64)[0]
        data = np.fromfile(fid, dtype=np.uint16).astype('float64')
        data = data.reshape((-1, channels))
    sample_period *= 1e-6 
    return sample_period, data

filename = 'Lab1/testBIns/test13.bin'
sample_period, data = raspi_import(filename)

signal_with_noise = data[:, 0]
sampling_rate = 1 / sample_period

frequency = 100 
time = np.arange(len(signal_with_noise)) / sampling_rate
pure_signal = np.sin(2 * np.pi * frequency * time)

noise_estimate = signal_with_noise - pure_signal

signal_rms = np.sqrt(np.mean(pure_signal ** 2))
noise_rms = np.sqrt(np.mean(noise_estimate ** 2))

snr_linear = signal_rms / noise_rms
snr_db = 20 * np.log10(snr_linear)

print(f"SNR (dB): {snr_db}")
