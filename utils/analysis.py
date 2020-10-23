import numpy as np
from scipy import ndimage
import itertools
import pylab
import scipy
from scipy import ndimage


#Thanks to Alex Krolewski for the basis of the this code!

def compute_deformation_tensor(dm_density):
	delta_k = np.fft.fftn(dm_density)
	kx = np.fft.fftfreq(len(delta_k))
	ky = np.fft.fftfreq(len(delta_k))
	kz = np.fft.fftfreq(len(delta_k))
	kx_3d,ky_3d,kz_3d = np.meshgrid(kx,ky,kz)
	ksq_3d = kx_3d**2 + ky_3d**2 + kz_3d**2

	t_kxkx = kx_3d*kx_3d*delta_k/ksq_3d
	t_kxky = kx_3d*ky_3d*delta_k/ksq_3d
	t_kxkz = kx_3d*kz_3d*delta_k/ksq_3d
	t_kyky = ky_3d*ky_3d*delta_k/ksq_3d
	t_kykz = ky_3d*kz_3d*delta_k/ksq_3d
	t_kzkz = kz_3d*kz_3d*delta_k/ksq_3d

	t_kxkx[0,0,0] = 0
	t_kxky[0,0,0] = 0
	t_kxkz[0,0,0] = 0
	t_kyky[0,0,0] = 0
	t_kykz[0,0,0] = 0
	t_kzkz[0,0,0] = 0

	t_xx = np.fft.ifftn(t_kxkx)
	t_xy = np.fft.ifftn(t_kxky)
	t_xz = np.fft.ifftn(t_kxkz)
	t_yy = np.fft.ifftn(t_kyky)
	t_yz = np.fft.ifftn(t_kykz)
	t_zz = np.fft.ifftn(t_kzkz)

	t_xx = np.real(t_xx)
	t_xy = np.real(t_xy)
	t_xz = np.real(t_xz)
	t_yy = np.real(t_yy)
	t_yz = np.real(t_yz)
	t_zz = np.real(t_zz)

	t_ij = np.array([[t_xx,t_xy,t_xz],[t_xy,t_yy,t_yz],[t_xz,t_yz,t_zz]])
	t_ij_trans = np.transpose(t_ij,axes=(2,3,4,0,1))

	e = np.linalg.eigh(t_ij_trans)
	return e

import numpy as np
import matplotlib.pyplot as plt

def make_hist_norm(x,binsize=10):
	hist, bins = np.histogram(x,bins=binsize)
	print(x.mean())    
	hist_norm = hist.astype(float)/float(np.sum(hist))
	hist_norm = list(hist_norm)
	hist_norm.append(hist_norm[-1])
	return bins, hist_norm



def eigen_stuff(truth, test, pad = 15, rebin_size=128,kernel_sizes = np.ones(4)*1.5,
color_pallete= ["r","b","g","v"]):
    

    #compute truth field quantities

    dm_density_bin = truth[pad:-1*pad,pad:-1*pad,pad:-1*pad]
    kernel_size = kernel_sizes[0]

    dm_density_smoothed = scipy.ndimage.filters.gaussian_filter(dm_density_bin,kernel_sizes[0]*(float(rebin_size)/64.0),mode='wrap')

    e_dm = compute_deformation_tensor(dm_density_smoothed)

    evals_dm_1a = e_dm[0][:,:,:,::-1]
    evecs_dm = e_dm[1]

    e1_dm_trutha = evecs_dm[:,:,:,:,0]
    e2_dm_trutha = evecs_dm[:,:,:,:,1]
    e3_dm_trutha = evecs_dm[:,:,:,:,2]

    eigentruth = [evals_dm_1a,e1_dm_trutha,e2_dm_trutha,e3_dm_trutha]
    eigenstuff = []
    for n,i in enumerate(test):
        print(n, i.shape)
        dm_density_bin = i[pad:-1*pad,pad:-1*pad,pad:-1*pad]
        kernel_size = kernel_sizes[n]

        # Mode=wrap means this is consistent with PBC
        dm_density_smoothed = scipy.ndimage.filters.gaussian_filter(dm_density_bin,kernel_sizes[n+1]*(float(rebin_size)/64.0),mode='wrap')

        e_dm = compute_deformation_tensor(dm_density_smoothed)

        evals_dm_4 = e_dm[0][:,:,:,::-1]
        evecs_dm = e_dm[1]

        e1_dm = evecs_dm[:,:,:,:,0]
        e2_dm = evecs_dm[:,:,:,:,1]
        e3_dm = evecs_dm[:,:,:,:,2]
        eigenstuff.append([evals_dm_4,e1_dm,e2_dm,e3_dm])

    return eigentruth,eigenstuff

def plot_eigenvectors(eigentruth,eigenstuff,
    color_pallete= ["r","b","g","v"],
    center_pallete = ["ro","bx","g*","v#"],
    labels = ["q","q1","q2"]):

#[ 
 #                 r'$\langle d_{\perp} \rangle$ = 1.0 (TARDIS)',
#                  r'$\langle d_{\perp} \rangle$ = 2.4 (TARDIS)',
#                  r'$\langle d_{\perp} \rangle$ = 3.7 (TARDIS)']

    plt.figure(figsize=(7,3))

    for ii in range(1,4):
  #  print(ii)
        ax1 = plt.subplot(int("13"+str(ii)))
        #[evals_dm_4,e1_dm,e2_dm,e3_dm]
        for nn,entry in enumerate(eigenstuff):
            print(ii,nn)
            bins, hist_norm = make_hist_norm(np.ndarray.flatten(np.abs(np.sum(entry[ii]*eigentruth[ii],axis=3))),binsize=10)
            plt.plot(bins,hist_norm,drawstyle='steps-post',color=color_pallete[nn])
            bin_centers = (bins[:-1]+bins[1:])/2.
            hist_centers = hist_norm[:-1]
            plt.plot(bin_centers,hist_centers,center_pallete[nn],ms=5)

        plt.xlabel(r'$\cos\theta$',size=12)

        plt.title(r'$\hat{e}_'+str(ii)+'$',size=16)
        plt.plot(0.1*np.ones(len(bins)),color='k',linestyle='--')
        ax1.set_xlim([0.0,0.99])
        ax1.set_ylim([0.01,1.0])
        plt.minorticks_on()
        plt.yscale('log')

        if ii>1:
            plt.setp(ax1.get_yticklabels(), visible=False)
        if ii==1:

            #Create custom artists
            a1 = plt.Line2D((0,1),(0,0), color='r', marker='o')
            a2 = plt.Line2D((0,1),(0,0), color='b', marker='x')
            a3 = plt.Line2D((0,1),(0,0), color='g', marker='*')
            pylab.rcParams['figure.figsize'] = (8.0, 4.0)

            #Create legend from custom artist/label lists
            ax1.legend([a1,a2,a3],
                      labels,loc=2,frameon=False,labelspacing=0.05,bbox_to_anchor=(-0.05,1.05),handletextpad=0.0,numpoints=1)

            #plt.legend(loc=2,frameon=False,labelspacing=0.15,bbox_to_anchor=(-0.05,1.05),handletextpad=0.0)

            leg = plt.gca().get_legend()
            ltext = leg.get_texts()
            plt.setp(ltext,fontsize='x-small')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    

def to_catalog(STATE, **kwargs):
    from nbodykit.lab import ArrayCatalog
    from nbodykit.transform import ConstantArray
    Omega = 0.27#Omega(self.a['S'])
    source = ArrayCatalog({'Position' : STATE[0,0], 'Velocity' : STATE[1,0],
            'Weight' : ConstantArray(Omega, len(STATE[0,0]))},
            BoxSize=[64,64,64], Nmesh=[64,64,64],Omega=Omega, Omega0=0.27,
            Time=0.228,  **kwargs
        )
    return source
    