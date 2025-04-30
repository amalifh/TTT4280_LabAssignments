import numpy as np
import matplotlib.pyplot as plt
import os

# === PARAMETERE (juster etter behov) ===
c = 3e8              # Lysfart (m/s)
fc = 24e9            # Radarfrekvens (Hz) - juster etter din sensor
sample_rate = 10000   # Samplerate (Hz) - juster etter ditt datasystem

# === FUNKSJON: Behandle én fil og returner målt hastighet ===
def analyze_file(filepath):
    data = np.fromfile(filepath, dtype=np.int16)
    I = data[0::2]
    Q = data[1::2]
    IQ = I + 1j * Q

    # FFT
    N = len(IQ)
    spectrum = np.fft.fftshift(np.fft.fft(IQ, n=N))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/sample_rate))
    velocity = freqs * c / (2 * fc)
    spectrum_magnitude = np.abs(spectrum)
    spectrum_dB = 20 * np.log10(spectrum_magnitude + 1e-12)

    # Finn hastighet med sterkest dopplerskift
    peak_idx = np.argmax(spectrum_magnitude)
    v_measured = velocity[peak_idx]

    # Plot dopplerspektrum
    plt.figure()
    plt.plot(velocity, spectrum_dB)
    plt.xlabel("Hastighet (m/s)")
    plt.ylabel("Amplitude (dB)")
    plt.title(f"Dopplerspektrum: {os.path.basename(filepath)}")
    plt.grid()
    plt.show()

    return v_measured

# === KJØR ANALYSE FOR FLERE FILER ===
# Legg til dine filer og tilhørende teoretiske hastigheter her
file_velocity_pairs = [
    ("Lab4/data/Speed-Slow/radar_slow1.bin", 1.6),
    # Legg gjerne til flere filer her:
    # ("/mnt/data/radar_medium1.bin", 3.0),
    # ("/mnt/data/radar_fast1.bin", 5.5),
]

theoretical = []
measured = []

for filepath, v_true in file_velocity_pairs:
    v_meas = analyze_file(filepath)
    theoretical.append(v_true)
    measured.append(v_meas)
    print(f"{os.path.basename(filepath)} -> Teoretisk: {v_true} m/s, Målt: {v_meas:.2f} m/s")

# === PLOTT målt vs teoretisk hastighet ===
plt.figure()
plt.plot(theoretical, measured, 'bo', label="Målt vs Teoretisk")
plt.plot(theoretical, theoretical, 'r--', label="Ideell linje (y=x)")
plt.xlabel("Teoretisk hastighet (m/s)")
plt.ylabel("Målt hastighet (m/s)")
plt.title("Målt vs Teoretisk Hastighet")
plt.legend()
plt.grid()
plt.show()
