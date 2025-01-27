import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline

# Step 1: Read the CSV file
data = pd.read_csv('lab1_riktigbode_lavpassfilter.csv')

# Step 2: Plot the data
# You can choose specific columns to plot
plt.plot(data['Frequency (Hz)'], data['Channel 2 Magnitude (dB)'])
#plt.plot(data['Time (s)'], data['Channel 2 (V)'], label='v_o')
plt.xlim(10,70)
plt.ylim(-20,10)

plt.axvline(x=31, color='red', linestyle='--', linewidth=1)

#plt.axhline(y=8.6,  color='red', linestyle='--', linewidth=1)
"""plt.axhline(y=-3, color='red', linestyle='--', linewidth=1)
plt.axhline(y=-10, color='red', linestyle='--', linewidth=1)

plt.

plt.plot(1763, -3.11, marker='o', markersize=7, color='green')
plt.plot(2350, -9.66, marker='o', markersize=7, color='green')"""

plt.xlabel('Frekvens [Hz]')
plt.ylabel('Amplituderespons [dB]')
plt.title('Bodediagram for lavpassfilter')

plt.grid(True)
#plt.legend()

# Step 3: Show the plot
#plt.show()

plt.savefig('Bodediagram_lavpassfilter.png', dpi = 700)