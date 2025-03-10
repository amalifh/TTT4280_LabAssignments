import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import detrend

#Must access the data, all 3 channels
def access_data(filename):
    with open(filename, 'r') as f:
        r = []
        g = []
        b = []

        lines = f.readlines()
        for i in range(3, len(lines)):
            line = lines[i].split(' ')
            r.append(float((line[0]).strip()))
            g.append(float((line[1]).strip()))
            b.append(float((line[2]).strip()))
    return r, g, b



if __name__ == "__main__":
    r, g, b =access_data('TTT4280_LabAssignments/Lab3/test_laan.txt')

    t = np.arange(0,30,30/len(r))

    print(len(r))
    print(len(t))

    plt.plot(t, r)
    plt.plot(t,g)
    plt.plot(t, b)
    #plt.xlim(0,0.1)
    plt.show()