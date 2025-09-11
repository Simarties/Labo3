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
plt.plot(cal[0,:],cal[1,:]) # on voit que les premiers pic sont en -200 et 200
plt.show()

#%%
picN, = np.where(cal[1,:] == np.max(cal[1,-200+midx: midx]))
picP, = np.where(cal[1,:] == np.max(cal[1,midx: midx+200]))
dist = (picP-picN)[0] #pixels
