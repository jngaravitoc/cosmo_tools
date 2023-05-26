"""
Various analysis routines for the orbital poles project in cosmological
simulations


dependecies:
  - scipy
  - Numpy
  - Astropy
  - Corrfunc


author: Nico Garavito-Camargo
github: jngaravitoc

"""

import numpy as np
from scipy import linalg

from astropy import units as u
from astropy.coordinates import Angle
import Corrfunc
from Corrfunc.mocks.DDtheta_mocks  import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf


import nba

class Analysis:
  def __init__(self, rmin, rmax):
    self.rmin = rmin
    self.rmax = rmax


  def poles_subhalos(self, subhalos, satellites=False):
    f = 1* (u.km/u.s).to(u.kpc/u.Gyr)
    dist = np.sqrt(np.sum(subhalos.d['pos']**2, axis=1))

    rcut = np.where((dist>self.rmin) & (dist<self.rmax))
                  
    subh_kin = nba.kinematics.Kinematics(subhalos.d['pos'][rcut], subhalos.d['vel'][rcut]*f)
    l, b = subh_kin.orbpole()
    lpol = Angle(l * u.deg)
    lpolw = lpol.wrap_at(360 * u.deg).degree  

    if satellites == True :
      dists = np.sqrt(np.sum(subhalos.s['pos']**2, axis=1))
      rcuts = np.where((dists>self.rmin) & (dists<self.rmax))
      subh_kin = nba.kinematics.Kinematics(subhalos.s['pos'][rcuts], subhalos.s['vel'][rcuts]*f)
      l, b = subh_kin.orbpole()
      lpol = Angle(l * u.deg)
      lpolw = lpol.wrap_at(360 * u.deg).degree  

      return lpolw, b

    else :
      return lpolw, b


  # Correlation function 

  def compute_2d_corrf(self, l, b, l2, b2, nbins=180, nthreads=1):
    """
    Compute angular correlation functions of l and b where l and b are defined: 
    l [0, 360]
    b [-90, 90]

    if l2 and b2 != 0 compute cross-correlation function between (l, b) and (l2, b2)


    returns 
    DD/RR - 1

    RR is computed analytically as: -N(N-1)*np.diff( cos(theta) )/2
    """


    npoints = len(l)
    # Setup the bins
    bins = np.linspace(0.1, 180.0, nbins + 1) # note the +1 to nbins

    if ((l2!=0) & (b2!=0)):
      autocorr=0
      DD_counts = DDtheta_mocks(autocorr, nthreads, bins,
                                RA1=l.astype(np.float32), DEC1=b.astype(np.float32), 
                                RA2=l2.astype(np.float32), DEC2=b2.astype(np.float32), ra_refine_factor=6, dec_refine_factor=6)
      RR_counts = -(npoints-1)*(np.diff(np.cos(bins*np.pi/180.))/2)
    else:
      autocorr=1
      DD_counts = DDtheta_mocks(autocorr, nthreads, bins,
                                RA1=l.astype(np.float32), DEC1=b.astype(np.float32), ra_refine_factor=6, dec_refine_factor=6)
      RR_counts = -npoints*(npoints-1)*(np.diff(np.cos(bins*np.pi/180.))/2)

    return bins[:-1], DD_counts['npairs']/RR_counts - 1


  def orbital_pole_dispersion(self, j_vec, pole=False, lmc_pole=None):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole or the LMC's pole.
    '''
    avg_j_vec = np.mean(j_vec, axis=0, dtype=np.float64)/np.linalg.norm(np.mean(j_vec, axis=0, dtype=np.float64), keepdims=True)
    avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec])

    # replace values outside the domain of arccos [-1,1] with 1 or -1
    # assuming they are near (within 0.01) -1 or 1, meaning totally aligned with reference pole
    # and using nanmean to ignore anything out of these bounds
    avg_j_dot_j = np.where((avg_j_dot_j>1)&(avg_j_dot_j<1.01), 1, avg_j_dot_j)
    avg_j_dot_j = np.where((avg_j_dot_j<-1)&(avg_j_dot_j>-1.01), -1, avg_j_dot_j)
    pole_disp = np.sqrt(np.nanmean(np.arccos(avg_j_dot_j, dtype=np.float64)**2, dtype=np.float64))
    pole_disp = np.degrees(pole_disp, dtype=np.float64)

    return pole_disp, avg_j_vec 


if __name__ == "__main__":
  
    snap_init = 300
    snap_final = 600
    tsteps = snap_final - snap_init
    nbins = 30
    #nbins = 60
    rmin=300
    rmax=600
    sim='m12c'
    auto = False
    sats = True
    
    wmatrix = np.zeros((tsteps, nbins))
    wmatrix_s = np.zeros((tsteps, nbins))
    
   
    sim_directory = "/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/".format(sim)
    lOP, bOP = poles_subhalos(snap_init, rmin, rmax, satellites=sats)

    snap_times = "/mnt/ceph/users/firesims/fire2/metaldiff/{}_res7100/snapshot_times.txt".format(sim)
    times = np.loadtxt(snap_times, usecols=3)
    
    if sim=='m12b':
      lsat, bsat = get_halo_satellite(sim, -2)
      ltimes = [times[385], times[449]]
      
    elif sim=='m12c':
      lsat, bsat = get_halo_satellite(sim, -4)
      ltimes = [times[549]]

    elif sim=='m12f':
      lsat, bsat = get_halo_satellite(sim, -4)
      ltimes = [times[320], times[462]]

    elif sim=='m12i':
      lsat, bsat = get_halo_satellite(sim, -11)
      ltimes = []

    elif sim=='m12m':
      lsat, bsat = get_halo_satellite(sim, -19)
      ltimes = [times[444], times[558]]

    elif sim=='m12r':
      lsat, bsat = get_halo_satellite(sim, -2)
      lsat2, bsat2 = get_halo_satellite(sim, -3)
      lsat3, bsat3 = get_halo_satellite(sim, -5)
      ltimes = [times[477], times[515], times[560]]

    elif sim=='m12w':
      lsat, bsat= get_halo_satellite(sim, -3)
      lsat2, bsat2= get_halo_satellite(sim, -7)
      lsat3, bsat3= get_halo_satellite(sim, -8)
      ltimes = [times[311], times[358], times[490]]



    bins, w0s = an.compute_2d_corrf(lOP, bOP, np.array([np.nanmean(lsat)]), np.array([np.nanmean(bsat)]), nbins)
    bins, w0 = an.compute_2d_corrf(lOP, bOP, np.array([0]), np.array([0]), nbins)
    wmatrix[0] = w0
    wmatrix_s[0] = w0s

    for k in range(snap_init+1, snap_final, 1):
        lOP, bOP = poles_subhalos(k, rmin, rmax, satellites=sats)
        # 300 is the snap @ where the lsat & bsat array starts
        bins, wmatrix_s[k-snap_init] = an.compute_2d_corrf(lOP, bOP, np.array([np.nanmean(lsat)]), 
                                                        np.array([np.nanmean(bsat)]), nbins)
        bins, wmatrix[k-snap_init] = an.compute_2d_corrf(lOP, bOP, np.array([0]), 
                                                        np.array([0]), nbins)
    if auto == True :
      np.savetxt('{}_wmatrix_corrfunc_sat_{}_{}_subhalos_sats_{}.txt'.format(sim, rmin, rmax, sats), wmatrix)
      plot_2dcorrfunc(wmatrix_s, 0, times[snap_init], times[snap_final],  r'${}$'.format(sim), '{}_2d_corrfunc_sat_{}_{}_{}.pdf'.format(sim, sats, rmin, rmax), ltimes, vmin=-2, vmax=2)
    else :
      np.savetxt('{}_wmatrix_corrfunc_{}_{}_subhalos_sats_{}.txt'.format(sim, rmin, rmax, sats), wmatrix)
      plot_2dcorrfunc(wmatrix, wmatrix[0], times[snap_init], times[snap_final],  r'${}$'.format(sim), '{}_2d_corrfunc_{}_{}_{}.pdf'.format(sim, sats, rmin, rmax), ltimes, vmin=-0.1, vmax=0.1)
