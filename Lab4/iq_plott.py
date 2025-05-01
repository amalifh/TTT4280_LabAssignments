import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

# --- Funksjon for å importere radarfil ---
def raspi_import(path, channels=2):
    with open(path, 'rb') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        data = data.reshape((-1, channels))
    sample_period *= 1e-6  # konverter til sekunder
    return sample_period, data

# --- Hovedprogram ---
if __name__ == "__main__":
    # Last inn data
    sample_period, data = raspi_import('Lab4/data/SSV/radar_veryslow5.bin')  # endre sti om nødvendig
    fs = 1 / sample_period  # samplerate

    # --- Konverter fra uint16 til volt ---
    delta = 0.0008  # 0.8 mV per bit
    zero_level = 32768  # midten av 16-bit unsigned ADC

    # Fjern DC og lineær trend, skaler til volt
    I = detrend((data[:, 0] - zero_level) * delta)
    Q = detrend((data[:, 1] - zero_level) * delta)

    plt.plot(I[len(I)//2-500:len(I)//2+500]+(np.abs(max(I[len(I)//2-500: len(I)//2+500])+min(I[len(I)//2-500:len(I)//2+500]))//2), Q[len(I)//2-500: len(I)//2+500]+(np.abs(max(Q[len(Q)//2-500:len(Q)//2+500])+min(Q[len(Q) //2-500:len(Q)//2+500]))//2), label='I-Q plot')
    plt.title("I-Q plot: ")
    plt.xlabel("Reell akse[mV]")
    plt.ylabel(" Imagin r akse[mV]")
    plt.show() #funker ikke:(
