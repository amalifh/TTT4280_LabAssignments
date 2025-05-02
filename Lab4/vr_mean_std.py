import numpy as np

c = 3*10**8
f_0 = 24*10**9

sakte_fD = [213,211,211,203,212]
fort_fD = [296,299,296,298,284]
neg_fD = [-348,-273,-336,-233,-257]

def vr(fD, f0):
    liste = []
    for i in fD:
        vr = (c*i)/(2*f0)
        liste.append(vr)
    return liste

vr_sakte = vr(sakte_fD, f_0)
mean_sakte = np.mean(vr_sakte)

vr_rask = vr(fort_fD, f_0)
mean_rask = np.mean(vr_rask)

vr_neg = vr(neg_fD, f_0)
mean_neg = np.mean(vr_neg)

def std(mean, values):
    counter = 0
    for i in range(len(values)):
        counter += (values[i]-mean)**2
    return (np.sqrt(1/(len(values)-1)*counter))

print("Sakte: ",mean_sakte, std(mean_sakte, vr_sakte))
print("Fort: ", mean_rask, std(mean_rask,vr_rask))
print("Negativ: ", mean_neg, std(mean_neg, vr_neg))


