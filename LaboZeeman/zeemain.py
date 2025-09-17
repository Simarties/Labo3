import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

#%%
#trouver les peaks et leurs distance
cal=np.loadtxt("donnees//labo3//ZEE_AR_NS//546nm//plo1_0.00A.csv",delimiter=',',skiprows=1).T
A0=np.loadtxt("donnees//labo3//ZEE_AR_NS//546nm//plo1_0.00A.csv",delimiter=',',skiprows=1).T
ONEPIECE = find_peaks(cal[1,:],height=40,prominence=0.05*np.ptp(cal[1,:]),distance=len(cal[1,:])//200)
plt.plot(cal[0,:],cal[1,:])
plt.scatter(ONEPIECE[0],ONEPIECE[1]['peak_heights'])
plt.show()

#prendre la valeur médiane en x et y ici par simple volonté de clarté pour la vue, ne change absolument rien au reste
midx = round(( cal[0,-1] - cal[0,0] )/2) # puisque x est un array de 0 à... sa valeur
                                         # est aussi la position de la valeur de y
#%% À cause de l'emplacement de la caméra, il n'y a pas le même nombre de pic à gauche et à droite

def équilibre(ary):
    arr = ary.copy()
    arr -= midx
    picP = arr[arr>0]
    picN = arr[arr<0]

    if len(picP) > len(picN):
        surplus = len(picP) - len(picN)
        detrop = np.argsort(-np.abs(picP))[:surplus]
        a_enlever = picP[detrop]
        mask = np.ones(len(arr),dtype=bool)
        for val in a_enlever:
            mask[np.where(arr==val)[0][0]] = False
        équi = arr[mask]

    elif len(picP) < len(picN):
        surplus = len(picN) - len(picP)
        detrop = np.argsort(-np.abs(picN))[:surplus]
        a_enlever = picN[detrop]
        mask = np.ones(len(arr),dtype=bool)
        for val in a_enlever:
            mask[np.where(arr==val)[0][0]] = False
        équi = arr[mask]

    else:
        équi = arr
    return équi + midx


def rayons(y):
    arr = équilibre(y)
    ray = arr[::-1]-arr
    return ray[ray>0]


R = rayons(ONEPIECE[0])[::-1]
r_n = [(i/2)**2 for i in R]
p = np.array([i+1 for i in range(len(r_n))])

def f(p,a,b):
    return a*p+b

popt,pcov = curve_fit(f,p,r_n)
print(f'pente est de: {popt[0]:.2e} +- {pcov[0,0]**(1/2):.0e}')


#calcul du f
lamb546 = 546e-09
n0_546 = 2*2e-03/lamb546
f546 = np.sqrt(popt[0]*n0_546/2)
df456 = 1/2 * np.sqrt(n0_546/(2*popt[0])) *pcov[0,0]**(1/2)

#%%
plt.scatter(p,r_n)
plt.plot(p,p*popt[0]+popt[1])
plt.xlabel('ordre d\'interférence p')
plt.ylabel(r'$r_n^2$ ($px^2$)')
plt.tight_layout()
plt.show()


#%% calcul des niveaux d'énergies
etat_i = [1,0,1] #nb atomiques s,l,j
etat_f546 = [1,1,2]
etat_f436 = [1,1,1]


def g_lande(s,l,j):
    return 1 + ( j*(j+1) + s*(s+1) - l*(l+1) )/(2*j*(j+1))

g_landé_i = g_lande(*etat_i)
g_landé_f546 = g_lande(*etat_f546)
g_landé_f436 = g_lande(*etat_f436)

def energy(si,li,ji,sf,lf,jf): #calcul les niveau d'énergie avec le champ B (sans compter la valeur de B ou du magnéton)
    g_landé_i = g_lande(si,li,ji)
    g_landé_f = g_lande(sf,lf,jf)

    Mi = [i for i in range(-ji,ji+1)]
    Mf = [i for i in range(-jf,jf+1)]

    if lf-li == 1 or lf-li == -1 :
        for mi in Mi:
            for mf in Mf:
                if mf - mi == 0 or mf - mi == 1 or mf - mi == -1:
                    print('psi_i',li,si,ji,mi,'et psi_f',lf,sf,jf,mf)
                    delta_i = g_landé_i*mi
                    delta_f = g_landé_f*mf

                    decalage = delta_f - delta_i
                    print(decalage)

#%% conversion de la résistance en champ B
def RtoB(R):
    return R*0.2823
R436 = np.loadtxt('donnees//labo3//ZEE_AR_NS//436nm//R436nm.csv',delimiter=',',skiprows=1)[:,1]
B436 = RtoB(R436)

#en partant de la théorie de le f est d'environ 12cm et que il y a 0.005cm/pixel,
