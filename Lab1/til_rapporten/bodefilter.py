import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline

# Step 1: Read the CSV file
data = pd.read_csv('Lab4/Bodeplot_filter.csv')

freq = data['Frequency (Hz)']
amp = data['Channel 2 Magnitude (dB)']
# Step 2: Plot the data
# You can choose specific columns to plot
plt.semilogx(freq, amp, label = 'Amplituderespons')
#plt.plot(data['Time (s)'], data['Channel 2 (V)'], label='v_o')
plt.xlim(1,4000)

plt.ylim(8,20.3)

#Legge til røde striplede linjer ved cutoff

peak = amp.max()
print(peak)
cutoff_level = peak - 3.0
print(cutoff_level)



plt.plot(3.82,17.18, 'ro') 
plt.plot(2629, 17.18, 'ro')

plt.axhline(cutoff_level, color='red', linestyle='--', label='-3 dB nivå') # red dot

plt.axvline(x=3.82, color='red', linestyle='--', linewidth=1)
plt.axvline(x=2629,color='red', linestyle='--', linewidth=1)

plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)

#plt.axhline(y=8.6,  color='red', linestyle='--', linewidth=1)
"""plt.axhline(y=-3, color='red', linestyle='--', linewidth=1)
plt.axhline(y=-10, color='red', linestyle='--', linewidth=1)

plt.

plt.plot(1763, -3.11, marker='o', markersize=7, color='green')
plt.plot(2350, -9.66, marker='o', markersize=7, color='green')"""

plt.xlabel('Frekvens [Hz]')
plt.ylabel('Amplituderespons [dB]')
plt.title('Bodediagram for aktivt båndpass med cutoff ved 3.8 Hz og 2.63 kHz', fontsize = 12)

plt.grid(True)
plt.legend()

# Step 3: Show the plot
plt.show()

#plt.savefig('Bodediagram_filter1_butterworth.png', dpi = 700)