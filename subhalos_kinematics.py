import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy import units as u
import sys
import pynbody
import healpy as hp
from healpy.newvisufunc import projview, newprojplot

sys.path.append("../scripts/src/")

sys.path.append("/mnt/home/ecunningham/python")
plt.style.use('~/matplotlib.mplstyle')
import gizmo_analysis as ga
import halo_analysis as halo
import nba

# 
import pynbody_routines as pr 
import io_gizmo_pynbody as fa
import plotting as pl

from scipy.linalg import norm



def orbital_pole_dispersion(pos, vel):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole.
    '''
    j_vec = np.cross(pos,vel)
    j_mag = norm(j_vec, axis=1)
    j_vec_norm = j_vec.T/j_mag
    #j_vec = orbital_ang_momentum(hal, hal_mask, host_str=host_str, norm=True)
    avg_j_vec = np.nanmean(j_vec_norm.T, axis=0, dtype=np.float64)/np.linalg.norm(np.nanmean(j_vec_norm.T, axis=0))
    #print(np.linalg.norm(np.nanmean(j_vec, axis=0)))
    avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec_norm.T]) 
    pole_disp = np.sqrt(np.nanmean(np.arccos(avg_j_dot_j)**2, dtype=np.float64))
    pole_disp = np.degrees(pole_disp)
    return pole_disp#, norm(avg_j_vec)
    
    
if __name__ ==  "__main__":
    sim=sys.argv[1]
    sim_directory = "/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/".format(sim)
    snap_times = "/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/snapshot_times.txt".format(sim)
    times = np.loadtxt(snap_times, usecols=3)
    m12i_nosat = fa.FIRE(sim, remove_satellite=True, rm_stellar_sat=True)
    m12i = fa.FIRE(sim)
    rcut = 50
    mcut = 1e7
    m12i_op_disp = np.zeros(300)
    m12i_op_disp_nosat = np.zeros(300)
    m12i_op_disp_stars = np.zeros(300)
    m12i_op_disp_nosat_stars = np.zeros(300)

    m12i_dark_median_vx = np.zeros(300)
    m12i_dark_median_vy = np.zeros(300)
    m12i_dark_median_vz = np.zeros(300)
    m12i_dark_std_vx = np.zeros(300)
    m12i_dark_std_vy = np.zeros(300)
    m12i_dark_std_vz = np.zeros(300)


    m12i_dark_median_x = np.zeros(300)
    m12i_dark_median_y = np.zeros(300)
    m12i_dark_median_z = np.zeros(300)
    m12i_stellar_median_x = np.zeros(300)
    m12i_stellar_median_y = np.zeros(300)
    m12i_stellar_median_z = np.zeros(300)

    m12i_stellar_median_vx = np.zeros(300)
    m12i_stellar_median_vy = np.zeros(300)
    m12i_stellar_median_vz = np.zeros(300)
    m12i_stellar_std_vx = np.zeros(300)
    m12i_stellar_std_vy = np.zeros(300)
    m12i_stellar_std_vz = np.zeros(300)

    
    for k in range(300, 600):
        sub_not_sat = m12i_nosat.subhalos(k)
        dsub_nosat = np.sqrt(np.sum(np.array(sub_not_sat.dark['pos'])**2, axis=1))
        dcut_nosat = np.where((dsub_nosat<300) & (dsub_nosat>rcut) & (sub_not_sat.dark['mass']>mcut))[0]

        dsub_nosat_stars = np.sqrt(np.sum(np.array(sub_not_sat.star['pos'])**2, axis=1))
        dcut_nosat_stars = np.where((dsub_nosat_stars<300) & (dsub_nosat_stars>rcut) & (sub_not_sat.star['mass']>0))[0]



        sub = m12i.subhalos(k)
        dsub = np.sqrt(np.sum(np.array(sub.dark['pos'])**2, axis=1))
        dcut = np.where((dsub<300) & (dsub>rcut) & (sub.dark['mass']>mcut))[0]
        dsub_stars = np.sqrt(np.sum(np.array(sub.star['pos'])**2, axis=1))
        dcut_stars = np.where((dsub_stars<300) & (dsub_stars>rcut) & (sub.star['mass']>0))[0]

        m12i_op_disp_nosat[k-300] = orbital_pole_dispersion(np.array(sub_not_sat.dark['pos'])[dcut_nosat], 
                                                            np.array(sub_not_sat.dark['vel'])[dcut_nosat])

        m12i_op_disp[k-300] = orbital_pole_dispersion(np.array(sub.dark['pos'])[dcut], 
                                                      np.array(sub.dark['vel'])[dcut])

        m12i_op_disp_nosat_stars[k-300] = orbital_pole_dispersion(np.array(sub_not_sat.star['pos'])[dcut_nosat_stars], 
                                                            np.array(sub_not_sat.star['vel'])[dcut_nosat_stars])

        m12i_op_disp_stars[k-300] = orbital_pole_dispersion(np.array(sub.star['pos'])[dcut_stars], 
                                                      np.array(sub.star['vel'])[dcut_stars])


        

        m12i_dark_median_vx[k-300] = np.nanmedian(sub_not_sat.dark['vel'][:,0]) 
        m12i_dark_median_vy[k-300] = np.nanmedian(sub_not_sat.dark['vel'][:,1])
        m12i_dark_median_vz[k-300] = np.nanmedian(sub_not_sat.dark['vel'][:,2])
        m12i_dark_median_x[k-300] = np.nanmedian(sub_not_sat.dark['pos'][:,0]) 
        m12i_dark_median_y[k-300] = np.nanmedian(sub_not_sat.dark['pos'][:,1])
        m12i_dark_median_z[k-300] = np.nanmedian(sub_not_sat.dark['pos'][:,2])
        m12i_dark_std_vx[k-300] = np.nanstd(sub_not_sat.dark['vel'][:,0])
        m12i_dark_std_vy[k-300] = np.nanstd(sub_not_sat.dark['vel'][:,1])
        m12i_dark_std_vz[k-300] = np.nanstd(sub_not_sat.dark['vel'][:,2])

        m12i_stellar_median_vx[k-300] = np.nanmedian(sub_not_sat.star['vel'][:,0]) 
        m12i_stellar_median_vy[k-300] = np.nanmedian(sub_not_sat.star['vel'][:,1])
        m12i_stellar_median_vz[k-300] = np.nanmedian(sub_not_sat.star['vel'][:,2])
        m12i_stellar_median_x[k-300] = np.nanmedian(sub_not_sat.star['pos'][:,0])
        m12i_stellar_median_y[k-300] = np.nanmedian(sub_not_sat.star['pos'][:,1])
        m12i_stellar_median_z[k-300] = np.nanmedian(sub_not_sat.star['pos'][:,2])
        m12i_stellar_std_vx[k-300] =np.nanmedian(sub_not_sat.star['vel'][:,0])
        m12i_stellar_std_vy[k-300] =np.nanmedian(sub_not_sat.star['vel'][:,1])
        m12i_stellar_std_vz[k-300] =np.nanmedian(sub_not_sat.star['vel'][:,2])
    
    results = np.array([times[300:600], m12i_op_disp_nosat, m12i_op_disp, m12i_op_disp_nosat_stars, m12i_op_disp_stars]).T
    median_v = np.array([times[300:600], m12i_dark_median_vx, m12i_dark_median_vy, m12i_dark_median_vz, m12i_dark_std_vx,
                        m12i_dark_std_vy, m12i_dark_std_vz]).T

    median_r = np.array([times[300:600], m12i_dark_median_x, m12i_dark_median_y, m12i_dark_median_z, m12i_stellar_median_x, m12i_stellar_median_y, m12i_stellar_median_z]).T

    median_v_st = np.array([times[300:600], m12i_stellar_median_vx, m12i_stellar_median_vy, m12i_stellar_median_vz, m12i_stellar_std_vx,
                        m12i_stellar_std_vy, m12i_stellar_std_vz]).T

    np.savetxt("{}_op_analysis_mass1e7.txt".format(sim), results)
    #np.savetxt("{}_op_median_vel_mass1e8.txt".format(sim), median_v)
    #p.savetxt("{}_op_median_pos_mass1e8.txt".format(sim), median_r)
    #p.savetxt("{}_op_median_vel_stellar_mass1e8.txt".format(sim), median_v_st)
