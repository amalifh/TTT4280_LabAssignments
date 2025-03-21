import numpy as np


muabo = np.genfromtxt("Lab3/muabo.txt", delimiter=",")
muabd = np.genfromtxt("Lab3/muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 520 # Replace with wavelength in nanometres
blue_wavelength = 470 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 1 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth
def delta (musA, musrR):
    return np.sqrt(1/(3*(musA*(musA+musrR))))

#implementing transmittans
def T (muaR,musRR):
    d = 300 * 10**(-6) #dybde i m
    C = np.sqrt(3*muaR*(muaR+musRR))
    return np.exp(-C*d)

def Reflektans (muaR,musRR):
    d = 0.01 #dybde i m
    C = np.sqrt(3*muaR*(muaR+musRR))
    return np.exp(-2*C*d)


print(delta(mua[0],musr[0]))
print(delta(mua[1],musr[1]))
print(delta(mua[2],musr[2]))

print (f'Transmittans for rød: {T(mua[0],musr[0])}')
print (f'Transmittans for grønn: {T(mua[1],musr[1])}')
print (f'Transmittans for blå: {T(mua[2],musr[2])}')
print (f'Reflektans for rød: {Reflektans(mua[0],musr[0])}')
print (f'Refelektans for rød: {Reflektans(mua[1],musr[1])}')
print (f'Reflektans for rød: {Reflektans(mua[2],musr[2])}')



print(f'mua: {mua}, musr: {musr}')


#print(mua)
#print(musr)
