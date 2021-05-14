# Script to join analytical and simulation contrast in a single plot
# Data is taken from a json field with the following data:
# - dvals (array of double): values of distance between gratings in meters
# - frequency (array of double): frequencies of moire fringes corresponding to dvals in meters^-1
# - contrast (array of double): contrasts corresponding to dvals

import numpy as np
import matplotlib.pyplot as plt
import json 

Cfactor = 1 # factor to multiply analytical contrast (1 or 2)

path_sim = "./Contrast_data/PGMI2_contrast.json"
path_th = "./Contrast_data/PGMI2_contrast_analytical.json"

with open(path_sim, 'rb') as fp:
    datasim = json.load(fp)
    
with open(path_th, 'rb') as fp:
    datath = json.load(fp)
    

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('D [mm]')
ax1.set_ylabel('Contrast', color=color)
ax1.plot(np.array(datasim['dvals'])*1e3, datasim['contrast'], 'o', color=color)
ax1.plot(np.array(datath['dvals'])*1e3, np.array(datath['contrast'])*Cfactor, '-', color=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel('Frequency [mm^-1]', color=color)
ax2.plot(np.array(datasim['dvals'])*1e3, np.array(datasim['frequency'])*1e-3, 'x', color=color)
ax2.plot(np.array(datath['dvals'])*1e3, np.array(datath['frequency'])*1e-3, '-', color=color)

fig.tight_layout()
plt.show()
