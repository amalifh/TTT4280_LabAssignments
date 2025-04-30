import numpy as np
import matplotlib.pyplot as plt

# === Filbane og parametre ===
filename = 'Lab4/data/Speed-Slow/radar_slow1.bin'  # <-- Endre til riktig fil
sample_rate = 1000  # Hz (juster etter systemet ditt)

# === Last inn og separer I og Q ===
data = np.fromfile(filename, dtype=np.int16)
I = data[0::2]
Q = data[1::2]
time = np.arange(len(I)) / sample_rate

# === Plot I og Q separat ===
plt.figure()
plt.plot(time, I, label='I (In-phase)')
plt.plot(time, Q, label='Q (Quadrature)')
plt.xlim(0,0.1)
plt.xlabel('Tid (s)')
plt.ylabel('Amplitude')
plt.title('Rådata: I- og Q-signaler')
plt.legend()
plt.grid()
plt.show()
