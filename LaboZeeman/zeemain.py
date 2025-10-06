import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress
import pandas as pd
import os


def f(p, a, b):
    return a * p + b

#%%
#trouver les peaks et leurs distance
def graphes(file):
    ONEPIECE = find_peaks(file['Y'],height=max(file['Y'])*.35,prominence=0.025*np.ptp(file['Y']),distance=len(file['Y'])//200)
    plt.plot(file['X'],file['Y'])
    plt.scatter(ONEPIECE[0],ONEPIECE[1]['peak_heights'],label='Pic d\'intensité',color='indianred')
    plt.legend(fontsize=14,loc='upper right')
    plt.xlabel('distance (pixels)',fontsize=14)
    plt.ylabel('Intensité (sans unités)',fontsize=14)
    plt.title('Pic d\'intensité en fonction des rayons du patron de diffraction', fontsize=14)
    X = np.array(file['X'])
    midx = round((X[-1] - X[0]) / 2)
    plt.show()

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


        return équi + midx

    def rayons(y):
        arr = équilibre(y)
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

    popt_a,pcov_a = curve_fit(f,p_a,r_n_a)
    #print(f'pente des a est de: {popt_a[0]:.2e} +- {pcov_a[0,0]**(1/2):.0e}')
    popt_b,pcov_b = curve_fit(f,p_b,r_n_b)
    #print(f'pente des b est de: {popt_b[0]:.2e} +- {pcov_b[0,0]**(1/2):.0e}')



    #calcul des epsilon
    lamb546 = 546e-09
    n0_546 = 2*2e-03/lamb546
    f546_a = np.sqrt(popt_a[0]*n0_546/2)
    df456_a = 1/2 * np.sqrt(n0_546/(2*popt_a[0])) *pcov_a[0,0]**(1/2)

    f546_b = np.sqrt(popt_b[0]*n0_546/2)
    df456_b = 1/2 * np.sqrt(n0_546/(2*popt_b[0])) *pcov_b[0,0]**(1/2)

    epsilon_a = popt_a[1]*n0_546/2/f546_a**2 +1
    epsilon_b = popt_b[1]*n0_546/2/f546_b**2 +1

    d_epsilon_a2 = ( (n0_546/(2*f546_a**2))**2*pcov_a[1,1] + (n0_546*popt_a[1]/f546_a**3)*df456_a**2 )
    print('d_epsilon_a:', d_epsilon_a2)
    d_epsilon_b2 = ((n0_546/(2*f546_b**2))**2*pcov_b[1,1] + (n0_546*popt_b[1]/f546_b**3)*df456_b**2 )
    print('d_epsilon_b2:', d_epsilon_b2)

    delta_nu = np.abs(epsilon_a-epsilon_b)/2/2e-03
    d_delta_nu = np.sqrt(1/(2*2e-03)*(np.abs(d_epsilon_a2+d_epsilon_b2)))
    print(d_delta_nu)
    return delta_nu, d_delta_nu

#%% calcul des niveaux d'énergies
etat_i = [1,0,1] #nb atomiques s,l,j
etat_f546 = [1,1,2]
etat_f436 = [1,1,1]


def g_lande(s,l,j):
    return 1 + ( j*(j+1) + s*(s+1) - l*(l+1) )/(2*j*(j+1))

g_landé_i = g_lande(*etat_i)
g_landé_f546 = g_lande(*etat_f546)
g_landé_f436 = g_lande(*etat_f436)

def energy(si,li,ji,sf,lf,jf): #calcul les niveau d'énergie avec le champ B
    g_landé_i = g_lande(si,li,ji)
    g_landé_f = g_lande(sf,lf,jf)

    Mi = [i for i in range(-ji,ji+1)]
    Mf = [i for i in range(-jf,jf+1)]

    if lf-li == 1 or lf-li == -1 :
        for mi in Mi:
            for mf in Mf:
                if mf - mi == 0 or mf - mi == 1 or mf - mi == -1:
                    #print(f'psi_i li:{li} si:{si} ji:{ji} mi:{mi} et psi_f lf:{lf} sf:{sf} jf:{jf} mf:{mf}')
                    delta_i = g_landé_i*mi
                    delta_f = g_landé_f*mf
                    #print(mf-mi)
                    decalage = delta_f - delta_i
                    #print(decalage)
#cette partie du code m'a permis de voir que le décalage maximal sera égal à 4 et donc que c'est la valeur
# que l'on prendrons pour le calcul du magnéton


#%% conversion de la résistance en champ B
def RtoB(R):
    return R*0.2823



#%% Calcul de la valeur du magnéton pour 546nm

i=0
A = np.loadtxt('donnees//labo3//ZEE_AR_NS//546nm//R546nm.csv', delimiter=',', skiprows=1)
delta_nu = np.zeros(len(A))
d_delta = np.zeros(len(A))
for item_name in os.listdir('donnees//labo3//ZEE_AR_NS//546nm'):
    if '.csv' and 'plot_' in item_name:
        if item_name.replace('plot_','').replace("A.csv",'').replace('0','') in f'{A[:,0]}.csv':

            LeFile = pd.read_csv(f"donnees//labo3//ZEE_AR_NS//546nm//{item_name}",sep=None,engine='python')
            delta_nu[i],d_delta[i] = graphes(LeFile)
            i+=1

            #creer une fonction des trucs tout en haut pour calculer les delta nu, les garder dans un tableau, puis faire

pcob, cov = curve_fit(f, RtoB(A[:,1]), delta_nu)
plt.errorbar(RtoB(A[:,1]),delta_nu,yerr=d_delta,color='steelblue',label=r'$\Delta \nu$',ls='',marker='o')
plt.plot(RtoB(A[:,1]),RtoB(A[:,1])*pcob[0]+pcob[1],color='darkorange',label='régression linéaire')
plt.legend(loc='upper left',fontsize=14)
plt.grid()
plt.xlabel(r'B(T)',fontsize=14)
plt.ylabel(r'$\Delta \nu$ (GHz)',fontsize=14)
plt.title(r"$\Delta \nu$" f' en fonction d\'un champ magnétique B \n pour un filtre de 546nm ',fontsize=14)
plt.savefig(f"donnees//labo3_546")
plt.show()


delta_E = delta_nu*6.62607015e-34*1e09 #le facteur 10^9 est pour convertir les GHz en Hz
D_delta_E = d_delta*6.62607015e-34*1e09
d_B = 0.2823 * 0.01

mu_B = delta_E/4/RtoB(A[:,1])
d_mu_B = np.sqrt((D_delta_E/(4*RtoB(A[:,1])))**2 + ((delta_E/(4*(RtoB(A[:,1])**2)))*d_B)**2)
print(mu_B,d_mu_B)
print(RtoB(A[:,1]))

#%%
def graphe2(file,pics):
    X = np.array(file['X'])
    midx = round((X[-1] - X[0]) / 2)

    arr = pics[0].copy()
    arr -= midx
    valP = (arr[arr>0])[:4] + midx
    valN = -1*np.array(sorted(np.abs(arr[arr<0])))[:4] +midx

    mid = np.mean((valN+valP)/2)

    R436_a = (arr[(arr+midx)>mid] +midx - mid)[::2]
    R436_b = (arr[(arr+midx)>mid] +midx - mid)[1::2]

    r_n_a436 = [(i)**2 for i in R436_a]
    r_n_b436 = [(i)**2 for i in R436_b]

    p436_a = np.array([i+1 for i in range(len(r_n_a436))])
    p436_b = np.array([i+1 for i in range(len(r_n_b436))])

    #plt.scatter(r_n_a436,p436_a)
    #plt.show()


    popt436a, pcov436a = curve_fit(f,p436_a,r_n_a436)
    popt436b, pcov436b = curve_fit(f,p436_b,r_n_b436)

    lambd436 =436e-09

    n0_436 = 2*2e-03/lambd436
    f436_a = np.sqrt(popt436a[0]*n0_436/2)
    df436_a = 1/2 * np.sqrt(n0_436/(2*popt436a[0])) *pcov436a[0,0]**(1/2)

    f436_b = np.sqrt(popt436b[0]*n0_436/2)
    df436_b = 1/2 * np.sqrt(n0_436/(2*popt436b[0])) *pcov436b[0,0]**(1/2)
    return n0_436,f436_a,df436_a,f436_b,df436_b,popt436a,pcov436a,popt436b,pcov436b

#%%
def epsilon(n0,f_a,df_a,f_b,df_b,popt_a,pcov_a,popt_b,pcov_b):
    #calcul du epsilon
    epsilon_a = popt_a[1]*n0/2/f_a**2 +1
    epsilon_b = popt_b[1]*n0/2/f_b**2 +1
    #print(epsilon_a,epsilon_b)

    d_epsilon_a2 = ( (n0/(2*f_a**2))**2*pcov_a[1,1] + (n0*popt_a[1]/f_a**3)*df_a**2 )
    #print('d_epsilon_a:', d_epsilon_a2)
    d_epsilon_b2 = ((n0/(2*f_b**2))**2*pcov_b[1,1] + (n0*popt_b[1]/f_b**3)*df_b**2 )
    #print('d_epsilon_b2:', d_epsilon_b2)



    delta_nu = np.abs(epsilon_a-epsilon_b)/2/2e-03
    d_delta_nu = np.sqrt(1/(2*2e-03)*(np.abs(d_epsilon_a2+d_epsilon_b2)))
    #print(d_delta_nu)
    return delta_nu, d_delta_nu


#%% pour 436nm technique légèrement différente à prendre à cause de la configuration donc technique différente ici

i=0
B = np.loadtxt('donnees//labo3//ZEE_AR_NS//436nmV2//R436nm.csv', delimiter=',', skiprows=1)
delta_nu436 = np.zeros(len(B))
d_delta436 = np.zeros(len(B))
for item_name in os.listdir('donnees//labo3//ZEE_AR_NS//436nmV2'):
    if '.csv' and 'plot_' in item_name:
        if item_name.replace('plot_','').replace("A.csv",'').replace('0','') in f'{B[:,0]}.csv':

            fichier = pd.read_csv(f"donnees//labo3//ZEE_AR_NS//436nmV2//{item_name}",sep=None,engine='python')
            picos = find_peaks(fichier['Y'], height=40, prominence=0.01 * np.ptp(fichier['Y']), distance=6)
            #plt.plot(fichier['X'], fichier['Y'])
            #plt.scatter(picos[0], picos[1]['peak_heights'])
            #plt.show()
            delta_nu436[i],d_delta436[i] = epsilon(*graphe2(fichier,picos))
            i+=1
print(delta_nu436)
pcob436, cov436 = curve_fit(f, RtoB(B[:,1]), delta_nu436)
plt.errorbar(RtoB(B[:,1]),delta_nu436,yerr=d_delta436,color='steelblue',label=r'$\Delta \nu$',ls='',marker='o')
plt.plot(RtoB(B[:,1]),RtoB(B[:,1])*pcob436[0]+pcob436[1],color='darkorange',label='régression linéaire')
plt.legend(loc='upper left',fontsize=14)
plt.grid()
plt.xlabel(r'B(T)',fontsize=14)
plt.ylabel(r'$\Delta \nu$ (GHz)',fontsize=14)
plt.title(r"$\Delta \nu$" f' en fonction d\'un champ magnétique B \n pour un filtre de 436nm ',fontsize=14)
plt.savefig(f"donnees//labo3_436")
plt.show()

#%%
delta_E436 = delta_nu436*6.62607015e-34*1e09 #le facteur 10^9 est pour convertir les GHz en Hz
D_delta_E436 = d_delta436*6.62607015e-34*1e09
d_B = 0.2823 * 0.01

mu_B436 = delta_E436/4/RtoB(B[:,1])
d_mu_B436 = np.sqrt((D_delta_E436/(4*RtoB(B[:,1])))**2 + ((delta_E436/(4*(RtoB(B[:,1])**2)))*d_B)**2)
print(mu_B436,d_mu_B436)
print(RtoB(B[:,1]))

#%% RAIE SUPPLÉMENTAIRE on voit la superposition de plusieurs raies

fichier = pd.read_csv(f"donnees//labo3//ZEE_AR_NS//577nm//plot_4.90A.csv",sep=None,engine='python')
picos = find_peaks(fichier['Y'], height=40, prominence=0.05 * np.ptp(fichier['Y']), distance=6)
plt.plot(fichier['X'], fichier['Y'])
plt.scatter(picos[0], picos[1]['peak_heights'])
plt.show()


#%%
#calcul de la finesse

# Cette fonction permet de calculer la finesse d'un fichier de pics csv
def calc_finesse(csv_path, halfwin=20):
    data = pd.read_csv(csv_path)
    x, y = data["X"].values, data["Y"].values

    # Détection des pics
    peaks, _ = find_peaks(y, height=0.3*np.max(y), prominence=0.1*np.ptp(y))
    if len(peaks) < 2:
        return np.nan

    # Lorentzienne
    def lorentz(xv, x0, gamma, A, B):
        return A * (gamma**2) / ((xv - x0)**2 + gamma**2) + B

    fwhm = []   #valeur à mi-hauteur
    for pk in peaks:
        i1, i2 = max(0, pk-halfwin), min(len(x), pk+halfwin+1)
        xf, yf = x[i1:i2], y[i1:i2]
        med = np.median(yf)
        try:
            p0 = [x[pk], 3.0, max(y[pk]-med, 1e-6), med]
            bounds = ([x[pk]-5, 0.05, 0.0, min(yf)],
                      [x[pk]+5, 100.0, 10*np.max(yf), max(yf)])
            popt, _ = curve_fit(lorentz, xf, yf, p0=p0, bounds=bounds, maxfev=20000)
            _, gamma, _, _ = popt
            fwhm.append(2*abs(gamma))
        except:
            pass

    if len(fwhm) < 2:
        return np.nan

    x_peaks = np.sort(x[peaks])
    fsr = np.diff(x_peaks)
    fwhm = np.array(fwhm)
    fwhm_pair = 0.5*(fwhm[:-1] + fwhm[1:])

    finesse_vals = fsr / fwhm_pair
    return np.nanmean(finesse_vals)

# Création d'un tableau avec tous les csv de la finesse
files = [
    "donnees\\labo3\\ZEE_AR_NS\\Finesse\\plot_2mm.csv",
    "donnees\\labo3\\ZEE_AR_NS\\Finesse\\plot_4mm.csv",
    "donnees\\labo3\\ZEE_AR_NS\\Finesse\\plot_6mm.csv",
    "donnees\\labo3\\ZEE_AR_NS\\Finesse\\plot_8mm.csv",
    "donnees\\labo3\\ZEE_AR_NS\\Finesse\\plot_10mm.csv"
]

# Ici, on calcule la finesse pour chacun des fichiers dans le tableau
finesses = []
for f in files:
    F = calc_finesse(f)
    finesses.append(F)
    print(f"{f} : finesse = {F:.3f}" if not np.isnan(F) else f"{f} : échec du calcul")

# Calcul de la moyenne des finesses calculées dans le bloc ci-dessus
finesses_valides = [f for f in finesses if not np.isnan(f)]

finesse_moy = np.mean(finesses_valides)
print(f"\nFinesse moyenne sur {len(finesses_valides)} fichiers = {finesse_moy:.3f}")
