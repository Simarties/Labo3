import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt

#%%
#trouver le 0
cal=np.loadtxt("donnees//labo3//ZEE_AR_NS//546nm//plot_calibration.csv",delimiter=',',skiprows=1).T
A0=np.loadtxt("donnees//labo3//ZEE_AR_NS//546nm//plo1_0.00A.csv",delimiter=',',skiprows=1).T
plt.plot(cal[0,:],cal[1,:])
plt.show()


#%%
#prendre la valeur médiane en x et y
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


#avec la taille des pixels, nous pourrions trouver la distance réelle entre les raies pour avoir un système d'unités

lamb546 = 546e-09
n0 = 2*2e-03/lamb546

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
