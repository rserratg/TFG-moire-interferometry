# Script to join analytical and simulation contrast in a single plot

import numpy as np
import matplotlib.pyplot as plt
import json

#path_sim = "./contrast_data/PGMI2_pushin2017_bichrom_sim.json"
#path_th = "./contrast_data/PGMI2_pushin2017_bichrom_analytical.json"

path_sim = "./contrast_data/PGMI3_sarenac2018_monochrom.json"
path_th = "./contrast_data/PGMI3_sarenac2018_monochrom_analytical.json"

#path_sim = "./contrast_data/PGMI3_sarenac2018_maxwell.json"
#path_th = "./contrast_data/PGMI3_sarenac2018_maxwell_analytical.json"

Cfactor = 1 # factor to multiply contrast (1 or 2)

with open(path_sim, "rb") as fp:
    datasim = json.load(fp)

with open(path_th, "rb") as fp:
    datath = json.load(fp)

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('D [mm]')
ax1.set_ylabel('Contrast', color=color)
ax1.plot(np.array(datasim['dvals'])*1e3, datasim['contrast'], 'o', color=color)
ax1.plot(np.array(datath['dvals'])*1e3, np.array(datath['contrast'])*Cfactor, '-', color=color)

ax2 = ax1.twinx()

color = 'tab:red'
ax2.set_ylabel(r"Frequency [$mm^{-1}$]", color=color)
ax2.plot(np.array(datasim["dvals"])*1e3, np.array(datasim["frequency"])*1e-3, "x", color=color)
ax2.plot(np.array(datath['dvals'])*1e3, np.array(datath['frequency'])*1e-3, '-', color=color)

fig.tight_layout()
plt.show()
