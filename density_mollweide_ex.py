"""
Master script to analyze the FIRE simulations for the orbital poles project
(github/jngaravitoc/poles_in_cosmos)


This script has been tested with sims: m12b, m12i

Main functionalities:
   - Make plots
    - Density plots of the DM and stellar distribution in several projections 
    - Mollweide plots of particles and subhalos positions in Galactocentric
      coordinates.
    - Mollweide plots of the orbital poles
   - Perform analysis
    - Correlation function analysis

Dependencies:
  - scipy
  - numpy 
  - Gizmo Analysis
  - Halo tools
  - pynbody
  - Astropy
  - nba 

Author: Nico Garavito-Camargo
Github: jngaravitoc

TODO:
- Remove satellite subhalos

"""

#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import sys
import pynbody

sys.path.append("/mnt/home/ecunningham/python")
#plt.style.use('~/matplotlib.mplstyle')
import gizmo_analysis as ga
import halo_analysis as halo
import nba

# 
import pynbody_routines  as pr 
import plotting as pl
from io_gizmo_pynbody  import FIRE
import analysis as an
import sys


plt.rcParams['font.size'] = 35



if __name__ == "__main__":
    
    
    sim = sys.argv[1]
    snap_init = int(sys.argv[2])
    snap_final = int(sys.argv[3])
    ptype = sys.argv[4]
    bmin = float(sys.argv[5])
    bmax = float(sys.argv[6])
    sats = False
    rmin = 50
    rmax = 300
    rotate = True
    k = snap_init

    snap_times = '/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/snapshot_times.txt'.format(sim)
    times = np.loadtxt(snap_times, usecols=3)
    #plot_type = 'cartesian_projection' # vr_mollweide, orbital_poles 
    #plot_type = 'vr_mollweide'#, orbital_poles 
    plot_type = 'rho_mollweide'

    #m12 = FIRE(sim)
    m12 = FIRE(sim)

        
    f = 1* (u.km/u.s).to(u.kpc/u.Gyr)

    if ptype == 'star': 
        # face on particle data halo
        hfaceon = m12.rotated_halo(k)
        pos = hfaceon.star['pos']
        vel = hfaceon.star['vel']*f
    elif ptype == 'dark':
        hfaceon = m12.rotated_halo(k, rotate=rotate)
        pos = hfaceon.dark['pos']
        vel = hfaceon.dark['vel']*f

    dist = np.sqrt(np.sum(pos**2, axis=1))
    dist_cut1 = np.where((dist > rmin) & (dist< rmax)) 
        

    kinematics1 = nba.kinematics.Kinematics(pos[dist_cut1],  vel[dist_cut1])

    if plot_type == "vr_mollweide":        
        pos_galactic = kinematics1.pos_cartesian_to_galactic()
        vel_galactic = kinematics1.vel_cartesian_to_galactic()
        
        figname1 = "../../plots/exploration/{}_vr_dark_satellite_faceon_{:03d}.png".format(sim, k)
        fig_title = "{} satellite vr stars; {}-{} kpc; t={:.2f} Gyr".format(sim, rmin, rmax, times[k])

        #print(lsat[k], bsat[k], vel_galactic[0])5
        pl.mollweide_projection(pos_galactic[0]*180/np.pi, pos_galactic[1]*180/np.pi, 0,0, title=fig_title, bmin=-50, bmax=50, nside=40, smooth=5, q=vel_galactic[0], figname=figname1)
    elif plot_type == "cartesian_projection":
        figname = "./test_{}_DM_stars_proection_300_600".format(sim, k)
        pl.multipanel_plot(hfaceon, hfaceon, satellite_faceon, k, sim, figname)

    elif plot_type == 'rho_mollweide':
        figname = "../plots/exploration/{}_rho_{}_faceon_{:03d}.png".format(sim, ptype, k)
        fig_title = "{} satellite rho {}; {}-{} kpc; t={:.2f}  Gyr".format(sim, ptype, 50, 300, times[k])
        pos_galactic = kinematics1.pos_cartesian_to_galactic()
        pl.mollweide_projection(pos_galactic[0]*180/np.pi, pos_galactic[1]*180/np.pi, 
                                0, 0, 
                                title=fig_title, bmin=500, bmax=1000, nside=40, smooth=5, figname=figname)
    
    elif plot_type == "poles_mollweide":
        figname = "../plots/exploration/outer_{}_OP_{}_faceon_no_sat_{:03d}.png".format(sim, ptype, k)
        fig_title = "{} {}-{} kpc; t={:.2f} Gyr".format(sim, rmin, rmax, times[k])
        opl, opb = kinematics1.orbpole()
        opl_sat, opb_sat = kin_sat.orbpole()
        opl_ms, opb_ms = kin_ms.orbpole()
        print(len(opl_ms))
        pl.mollweide_projection(opl, opb, 
                                opl_sat[sat_pop2], opb_sat[sat_pop2],
                                l3=opl_sat[sat_pop3], b3=opb_sat[sat_pop3],
                                l4=opl_ms[snap_init-300], b4=opb_ms[snap_init-300],
                                title=fig_title, bmin=bmin, bmax=bmax, nside=40, smooth=1, figname=figname, cmap='Greys')
