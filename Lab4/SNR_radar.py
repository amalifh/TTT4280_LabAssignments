
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.fft import fft, fftfreq

# --- Funksjon for å importere radarfil ---
def raspi_import(path, channels=2):
    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        data = data.reshape((-1, channels))
    sample_period *= 1e-6  # konverter til sekunder
    return sample_period, data

# --- Hoveddel ---
if __name__ == "__main__":
    sample_period, data = raspi_import(sys.argv[1] if len(sys.argv) > 1 else 'Lab4/data/SR/radar_rev4.bin')

    # --- Detrend og lag komplekst signal ---
    I_signal = detrend(data[:, 0])
    Q_signal = detrend(data[:, 1])
    cmpx_signal = I_signal + 1j * Q_signal

    # --- FFT ---
    fft_data = np.fft.fftshift(np.fft.fft(cmpx_signal))
    n = len(cmpx_signal)
    frekvenser = np.fft.fftshift(np.fft.fftfreq(n, d=sample_period))

    # --- Beregn effekt og SNR ---
   # power_spectrum = np.abs(fft_data) ** 2
    #nyttig_spektrum = power_spectrum[:n // 2]
    #frekvenser_pos = frekvenser[:n // 2]

    # Signalstyrke = sterkeste topp
#    signal_index = np.argmax(nyttig_spektrum)
 #   P_signal = nyttig_spektrum[signal_index]

    # Velg intervall for støygulv (juster etter ditt signal)
    noise_f_min = 250  # Hz
    noise_f_max = 400  # Hz

    # Lag en maske for frekvensområdet
    #noise_mask = (frekvenser_pos >= noise_f_min) & (frekvenser_pos <= noise_f_max)

    # Beregn støygulv i det valgte området
   # P_noise = np.mean(nyttig_spektrum[noise_mask])


    # SNR i dB
  #  snr_linear = P_signal / P_noise
   # snr_db = 10 * np.log10(snr_linear)

    # --- Visualisering ---
    magnitude = np.abs(fft_data)
    magnitude_db = 20 * np.log10(magnitude)
    magnitude_db_normalisert = magnitude_db - np.max(magnitude_db)

    plt.figure(figsize=(10, 6))
    plt.plot(frekvenser, magnitude_db)
    #plt.plot(86.84, -35.45, 'o', color='indigo', label='Signaltopp på $86.84$ Hz')
    #plt.axhline(y=20 * np.log10(np.sqrt(P_noise)) - np.max(magnitude_db), color='salmon', linestyle='--', label='Støygulv')

    plt.xlabel('Frekvens [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.title('Frekvensspekter: Lav hastighet, SNR = $69.71$ dB')
    plt.grid()
    plt.xlim(-300, 300)
    #plt.ylim(-100, 0)
    plt.legend()
    plt.show()

    #print(f"SNR: {snr_db:.2f} dB")

    #plt.savefig("frekvenssprekter_lav_2.0.png", dpi = 700)

