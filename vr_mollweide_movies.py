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
    remove_satellite = True

    snap_times = '/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/snapshot_times.txt'.format(sim)
    times = np.loadtxt(snap_times, usecols=3)
    #plot_type = 'cartesian_projection' # vr_mollweide, orbital_poles 
    #plot_type = 'vr_mollweide'#, orbital_poles 
    plot_type = 'poles_mollweide' 
    
    #m12 = FIRE(sim)
    m12 = FIRE(sim, remove_satellite=remove_satellite)

    for k in range(snap_init, snap_final, 1):
        subhalos_faceon, satellite_faceon = m12.subhalos_rotated(k)
        #subhalos_satellites = m12.subhalos(k)
        
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
        elif ptype == 'subhalos':
            #pos = subhalos_satellites.star['pos']
            #vel = subhalos_satellites.star['vel']*f
            pos = subhalos_faceon.star['pos']
            vel = subhalos_faceon.star['vel']*f

        dist = np.sqrt(np.sum(pos**2, axis=1))

        dist_cut1 = np.where((dist > rmin) & (dist< rmax)) 
        #dist_cut2 = np.where((dist> 300) & (dist< 600)) 
        

        kinematics1 = nba.kinematics.Kinematics(pos[dist_cut1],  vel[dist_cut1])
        #kinematics2 = nba.kinematics.Kinematics(pos[dist_cut2],  vel[dist_cut2])
        
        
        kin_sat = nba.kinematics.Kinematics(subhalos_faceon.dark['pos'], subhalos_faceon.dark['vel']*f)
        lsat, bsat = kin_sat.pos_cartesian_to_galactic()
        mdark_sat = subhalos_faceon.dark['mass']
        sort_mass = np.sort(mdark_sat)
        pos_sat = np.sqrt(np.sum(subhalos_faceon.dark['pos']**2, axis=1))
        
        sat_pop1 = np.where((pos_sat>20) & (pos_sat<300))
        sat_pop2 = np.where((pos_sat>20) & (pos_sat<300) & (mdark_sat>1e8))
        sat_pop3 = np.where((pos_sat>20) & (pos_sat<300) & (mdark_sat>8e8))
        print(len(sat_pop1[0]), len(sat_pop2[0]), len(sat_pop3[0]))
        
        kin_ms = nba.kinematics.Kinematics(satellite_faceon.dark['pos'], satellite_faceon.dark['vel']*f)
        if plot_type == "vr_mollweide":        
            pos_galactic = kinematics1.pos_cartesian_to_galactic()
            vel_galactic = kinematics1.vel_cartesian_to_galactic()
            
            pos_galactic2 = kinematics2.pos_cartesian_to_galactic()
            vel_galactic2 = kinematics2.vel_cartesian_to_galactic()

            figname1 = "../../plots/exploration/{}_vr_dark_satellite_faceon_{:03d}.png".format(sim, k)
            fig_title = "{} satellite vr stars; {}-{} kpc; t={:.2f} Gyr".format(sim, rmin, rmax, times[k])

            #print(lsat[k], bsat[k], vel_galactic[0])5
            pl.mollweide_projection(pos_galactic[0]*180/np.pi, pos_galactic[1]*180/np.pi, lsat[:k-300+1]*180/np.pi, bsat[:k-300+1]*180/np.pi, title=fig_title, bmin=-50, bmax=50, nside=40, smooth=5, q=vel_galactic[0], figname=figname1)

            #figname2 = "../../plots/exploration/{}_vb_stars_satellite_faceon_{:03d}.png".format(sim, k)
            #fig_title2 = "{} satellite $v_{b}$ stars; {}-{} kpc; t={:.2f}  Gyr".format(sim, rmin, rmax, times[k])
            
            #pl.mollweide_projection(pos_galactic[0]*180/np.pi, pos_galactic[1]*180/np.pi, lsat[:k-snap_init]*180/np.pi, bsat[:k-snap_init]*180/np.pi, title=fig_title2, bmin=-50, bmax=50, nside=40, smooth=1, q=vel_galactic[1], figname=figname2)


            #figname2 = "../../plots/exploration/{}_vl_stars_satellite_faceon_{:03d}.png".format(sim, k)
            #fig_title2 = "{} satellite $v_{l}$ stars; {}-{} kpc; t={:.2f}  Gyr".format(sim, rmin, rmax, times[k])
            
            #pl.mollweide_projection(pos_galactic[0]*180/np.pi, pos_galactic[1]*180/np.pi, lsat[:k-snap_init]*180/np.pi, bsat[:k-snap_init]*180/np.pi, title=fig_title2, bmin=-50, bmax=50, nside=40, smooth=1, q=vel_galactic[1], figname=figname2)


            #figname1 = "../../plots/exploration/{}_vr_stars_satellite_faceon_300_600_{:03d}.png".format(sim, k)
            #fig_title = "{} satellite vr stars; {}-{} kpc; t={:.2f}  Gyr".format(sim, 300, 600, times[k])
            #pl.mollweide_projection(pos_galactic2[0]*180/np.pi, pos_galactic2[1]*180/np.pi, lsat[:k-snap_init]*180/np.pi, bsat[:k-snap_init]*180/np.pi, title=fig_title, bmin=-50, bmax=50, nside=40, smooth=1, q=vel_galactic2[0], figname=figname1)

        elif plot_type == "cartesian_projection":
            figname = "../plots/exploration/{}_DM_stars_projection_300_600".format(sim, k)
            pl.multipanel_plot(hfaceon, hfaceon, satellite_faceon, k, sim, figname)

        elif plot_type == 'rho_mollweide':
            figname = "../plots/exploration/{}_rho_{}_faceon_{:03d}.png".format(sim, ptype, k)
            fig_title = "{} satellite rho {}; {}-{} kpc; t={:.2f}  Gyr".format(sim, ptype, 50, 300, times[k])
            pos_galactic = kinematics1.pos_cartesian_to_galactic()
            print(len(pos_galactic[0]))
            pl.mollweide_projection(pos_galactic[0]*180/np.pi, pos_galactic[1]*180/np.pi, 
                                    lsat[k-300]*180/np.pi, 
                                    bsat[k-300]*180/np.pi, 
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
