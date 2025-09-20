import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import pandas as pd
import os

#%%
#trouver les peaks et leurs distance
def graphes(file):
    ONEPIECE = find_peaks(file['Y'],height=max(file['Y'])*.35,prominence=0.025*np.ptp(file['Y']),distance=len(file['Y'])//200)
    #plt.plot(file['X'],file['Y'])
    #plt.scatter(ONEPIECE[0],ONEPIECE[1]['peak_heights'])
    X = np.array(file['X'])
    midx = round((X[-1] - X[0]) / 2)
    #plt.show()

    # À cause de l'emplacement de la caméra, il n'y a pas le même nombre de pic à gauche et à droite
    # donc je crée une fonction qui s'arrange de prendre le même nombre de pic de chaque côté
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

            '''while len(équi)>20:
            m = 2
            trop = np.argsort(-np.abs(équi))[:m]
            enlever = équi[trop]
            mask2 = np.ones(len(équi),dtype=bool)
            for val in enlever:
                mask2[np.where(équi==val)[0][0]] = False
            équi = équi[mask2]'''




        return équi + midx

    def rayons(y):
        arr = équilibre(y)
        print(len(arr))
        ray = arr[::-1]-arr
        return ray[ray>0]

    R = rayons(ONEPIECE[0])[::-1]/2 #calcul tout les rayons, «intérieur» et «extérieur»

    R_a = R[::2]
    R_b = R[1::2]

    r_n = [(i)**2 for i in R]
    r_n_a = [(i)**2 for i in R_a]
    r_n_b = [(i)**2 for i in R_b]
    p = np.array([i+1 for i in range(len(r_n))])

    p_a = np.array([i+1 for i in range(len(r_n_a))])
    p_b = np.array([i+1 for i in range(len(r_n_b))])




    def f(p,a,b):
        return a*p+b

    #popt,pcov = curve_fit(f,p,r_n)
    #print(f'pente est de: {popt[0]:.2e} +- {pcov[0,0]**(1/2):.0e}')

    popt_a,pcov_a = curve_fit(f,p_a,r_n_a)
    print(f'pente des a est de: {popt_a[0]:.2e} +- {pcov_a[0,0]**(1/2):.0e}')

    popt_b,pcov_b = curve_fit(f,p_b,r_n_b)
    print(f'pente des b est de: {popt_b[0]:.2e} +- {pcov_b[0,0]**(1/2):.0e}')


    #calcul des epsilon
    #def eps(lambd,dist,pente_a,pente_b):

    lamb546 = 546e-09
    n0_546 = 2*2e-03/lamb546
    f546_a = np.sqrt(popt_a[0]*n0_546/2)
    df456_a = 1/2 * np.sqrt(n0_546/(2*popt_a[0])) *pcov_a[0,0]**(1/2)

    f546_b = np.sqrt(popt_b[0]*n0_546/2)
    df456_b = 1/2 * np.sqrt(n0_546/(2*popt_b[0])) *pcov_b[0,0]**(1/2)

    #calcul du epsilon
    epsilon_a = popt_a[1]*n0_546/2/f546_a**2 +1
    epsilon_b = popt_b[1]*n0_546/2/f546_b**2 +1

    delta_nu = np.abs(epsilon_a-epsilon_b)/2/2e-03
    return delta_nu
#%%
'''print(f'la distance focale est de {f546:.3e} +- {df456:.0e}')

#calcul des theta_n
theta_n = R/f546
print(theta_n)


#%%
plt.scatter(p,r_n)
plt.plot(p,p*popt[0]+popt[1])
plt.xlabel('ordre d\'interférence p')
plt.ylabel(r'$r_n^2$ ($px^2$)')
plt.tight_layout()
plt.show()
'''

#%% calcul des niveaux d'énergies
etat_i = [1,0,1] #nb atomiques s,l,j
etat_f546 = [1,1,2]
etat_f436 = [1,1,1]


def g_lande(s,l,j):
    return 1 + ( j*(j+1) + s*(s+1) - l*(l+1) )/(2*j*(j+1))

g_landé_i = g_lande(*etat_i)
g_landé_f546 = g_lande(*etat_f546)
g_landé_f436 = g_lande(*etat_f436)

def energy(si,li,ji,sf,lf,jf): #calcul les niveau d'énergie avec le champ B (sans compter la R546nm.csv de B ou du magnéton)
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


#%%

i=0
A = np.loadtxt('donnees//labo3//ZEE_AR_NS//546nm//R546nm.csv', delimiter=',', skiprows=1)
delta_nu = np.zeros(len(A))
for item_name in os.listdir('donnees//labo3//ZEE_AR_NS//546nm'):
    if '.csv' and 'plot_' in item_name:
        if item_name.replace('plot_','').replace("A.csv",'').replace('0','') in f'{A[:,0]}.csv':

            print(item_name)
            LeFile = pd.read_csv(f"donnees//labo3//ZEE_AR_NS//546nm//{item_name}",sep=None,engine='python')
            delta_nu[i] = graphes(LeFile)
            i+=1

            #creer une fonction des trucs tout en haut pour calculer les delta nu, les garder dans un tableau, puis faire
plt.scatter(RtoB(A[:,1]),delta_nu)
plt.show()