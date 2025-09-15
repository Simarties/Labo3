import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#%%
#trouver le 0
cal=np.loadtxt("donnees//labo3//ZEE_AR_NS//546nm//plot_calibration.csv",delimiter=',',skiprows=1).T
A0=np.loadtxt("donnees//labo3//ZEE_AR_NS//546nm//plo1_0.00A.csv",delimiter=',',skiprows=1).T
plt.plot(cal[0,:],cal[1,:])
plt.show()


#%%
#prendre la valeur médiane en x et y ici par simple volonté de clarté pour la vue, ne change absolument rien au reste
midx = round(( cal[0,-1] - cal[0,0] )/2) # puisque x est un array de 0 à... sa valeur
                                         # est aussi la position de la valeur de y
plt.plot(cal[0,:]-midx,cal[1,:]) # on voit que les premiers pic sont en -200 et 200
plt.show()

#%%
picN1, = np.where(cal[1,:] == np.max(cal[1,-200+midx: midx]))
picP1, = np.where(cal[1,:] == np.max(cal[1,midx: midx+200]))
dist1 = (picP1-picN1)[0] #pixels

picN2, = np.where(cal[1,:] == np.max(cal[1,-300+midx:-200+midx]))
picP2, = np.where(cal[1,:] == np.max(cal[1,midx+200:midx+300]))
dist2 = (picP2-picN2)[0] #pixels

picN3, = np.where(cal[1,:] == np.max(cal[1,-350+midx:-300+midx]))
picP3, = np.where(cal[1,:] == np.max(cal[1,midx+300:midx+400]))
dist3 = (picP3-picN3)[0]

picN4, = np.where(cal[1,:] == np.max(cal[1,-400+midx:-350+midx]))
picP4, = np.where(cal[1,:] == np.max(cal[1,400+midx:450+midx]))
dist4 = (picP4-picN4)[0]


#%% trouver la valeur de f en pixels

R = [dist1,dist2,dist3,dist4]
r_n = [(i/2)**2 for i in R]
p = np.array([i+1 for i in range(len(r_n))])

def f(p,a,b):
    return a*p+b

popt,pcov = curve_fit(f,p,r_n)
print(f'pente est de: {popt[0]:.2e} +- {pcov[0,0]**(1/2):.0e}')

#%%
plt.scatter(p,r_n)
plt.plot(p,p*popt[0]+popt[1])
plt.show()

#%% la pente correspond à 2f^2/n0

lamb546 = 546e-09
n0_546 = 2*2e-03/lamb546
f546 = np.sqrt(popt[0]*n0_546/2)
df456 = 1/2 * np.sqrt(n0_546/(2*popt[0])) *pcov[0,0]**(1/2)

#si possible aller trouver la distance focale au lab pour pouvoir trouver les distance theta

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
