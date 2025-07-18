from .math_module import xp, xcipy, ensure_np_array
from esc_psf import utils # SFR- Changed from esc_llowfsc (think this was an error?)
import esc_psf.props as props

import numpy as np
import astropy.units as u
import copy
import poppy
from scipy.signal import windows

import ray

from matplotlib.colors import LogNorm

class single():

    def __init__(
            self,
            wavelength_c=650e-9,
            wavelength=None,  
            total_pupil_diam=2.43,
            fsm_beam_diam=9.351e-3,
            dm_beam_diam=9.351e-3,
            lyot_beam_diam=4.153e-3,
            lyot_diam=3.7e-3,
            rls_diam=25.4e-3,
            llowfsc_defocus=5e-3,
            scc_position=(1.55*3.7e-3/np.sqrt(2), 1.55*3.7e-3/np.sqrt(2)),
            scc_diam=100e-6,
            Nact=34,
            act_spacing=300e-6,
            act_coupling=0.15,
            dm_ref=None, 
            camsci_pxscl=3.76e-6,
            camlo_pxscl=3.76e-6,
            camsci_fl=140e-3,
            camlo_fl=200e-3,
            entrance_flux=None, 
        ):
        
        self.wavelength_c = wavelength_c
        self.total_pupil_diam = total_pupil_diam
        self.fsm_beam_diam = fsm_beam_diam
        self.dm_beam_diam = dm_beam_diam
        self.lyot_pupil_diam = lyot_beam_diam
        self.lyot_diam = lyot_diam
        self.lyot_ratio = self.lyot_diam/self.lyot_pupil_diam
        self.rls_diam = rls_diam
        self.scc_position = scc_position
        self.scc_diam = scc_diam
        self.camsci_fl = camsci_fl
        self.camlo_fl = camlo_fl
        self.llowfsc_fnum = self.camlo_fl/self.lyot_diam
        self.llowfsc_defocus = llowfsc_defocus
        self.camsci_pxscl = camsci_pxscl
        self.camsci_pxscl_lamDc = self.camsci_pxscl / (self.camsci_fl * self.wavelength_c / self.lyot_pupil_diam)
        self.camlo_pxscl = camlo_pxscl
        self.camlo_pxscl_lamDc = self.camlo_pxscl / (self.camlo_fl * self.wavelength_c / self.lyot_pupil_diam)

        self.wavelength = wavelength if wavelength is not None else wavelength_c
        self.use_vortex = False
        self.plot_vortex = False
        self.plot_oversample = 1.5
        
        self.npix = 1000
        self.npix_rls = int( np.round( self.npix * self.rls_diam / self.lyot_pupil_diam ))
        self.def_oversample = 2.5 # default oversample
        self.rls_oversample = 6.5 # reflective lyot stop oversample
        self.Ndef = int(self.npix*self.def_oversample)
        self.Nscc = int(self.npix*self.def_oversample)
        self.Nrls = int(self.npix*self.rls_oversample)
        self.ncamsci = 500
        self.ncamlo = 96

        self.tt_pv_to_rms = 1/4
        self.as_per_radian = 206264.806

        ### INITIALIZE APERTURES ###
        pwf = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2 * u.m, npix=self.npix, oversample=self.def_oversample)
        self.APERTURE = poppy.CircularAperture(radius=self.lyot_pupil_diam/2 * u.m).get_transmission(pwf)
        self.BAP_MASK = self.APERTURE>0
        self.LYOTSTOP = poppy.CircularAperture(radius=self.lyot_ratio * self.lyot_pupil_diam/2 * u.m).get_transmission(pwf)
        
        self.use_scc = False
        self.SCC_PINHOLE = poppy.CircularAperture(radius=self.scc_diam * u.m, shift_x=self.scc_position[0]*u.m, shift_y=self.scc_position[1]*u.m).get_transmission(pwf)
        self.LYOTSCC = utils.pad_or_crop(self.LYOTSTOP, self.Nscc) + self.SCC_PINHOLE

        pwf_rls = poppy.FresnelWavefront(beam_radius=self.lyot_pupil_diam/2 * u.m, npix=self.npix, oversample=self.rls_oversample)
        rls_ap = poppy.CircularAperture(radius=self.rls_diam/2 * u.m).get_transmission(pwf_rls)
        self.RLS = rls_ap - utils.pad_or_crop( self.LYOTSTOP, self.Nrls)
        rls_ap = 0

        self.N = self.Ndef # default to not using RLS

        # Initialize pupil data
        self.PREFPM_AMP = xp.ones((self.npix,self.npix))
        self.PREFPM_OPD = xp.zeros((self.npix,self.npix))
        self.POSTFPM_AMP = xp.ones((self.npix,self.npix))
        self.POSTFPM_OPD = xp.zeros((self.npix,self.npix))
        self.RLS_AMP = xp.ones((self.Nrls,self.Nrls))
        self.RLS_OPD = xp.zeros((self.Nrls,self.Nrls))

        self.PTT_MODES = utils.create_zernike_modes(self.APERTURE, nmodes=3, remove_modes=0) # define tip/tilt modes
        self.FSM_PTT = np.zeros(3) # [OPD in m, arcsec, arcsec]
        self.FSM_OPD = 0*self.PTT_MODES[0]

        # Initialize flux and normalization params
        self.Imax_ref = 1
        self.entrance_flux = entrance_flux
        if self.entrance_flux is not None:
            ep_pixel_area = (self.total_pupil_diam*u.m / self.npix)**2
            flux_per_ep_pixel = self.entrance_flux * ep_pixel_area
            # print(type(self.APERTURE))
            # print(type(flux_per_ep_pixel.to_value(u.photon/u.second)))
            self.APERTURE *= np.sqrt(flux_per_ep_pixel.to_value(u.photon/u.second))

        ### INITIALIZE DM PARAMETERS ###
        self.Nact = Nact
        self.dm_shape = (self.Nact, self.Nact)
        self.act_spacing = act_spacing
        self.dm_pxscl = self.dm_beam_diam / self.npix
        self.inf_sampling = self.act_spacing / self.dm_pxscl
        self.inf_fun = utils.make_gaussian_inf_fun(
            act_spacing=self.act_spacing, 
            sampling=self.inf_sampling, 
            coupling=act_coupling, 
            Nact=self.Nact+2,
        )
        self.Nsurf = self.inf_fun.shape[0]

        y,x = (xp.indices((self.Nact, self.Nact)) - self.Nact//2 + 1/2)
        r = xp.sqrt(x**2 + y**2)
        self.dm_mask = r<(self.Nact/2 + 1/2)
        self.Nacts = int(self.dm_mask.sum())

        self.dm_ref = dm_ref if dm_ref is not None else xp.zeros((self.Nact,self.Nact))
        self.dm_channels = xp.zeros((10, self.Nact, self.Nact))
        self.dm_channels[0] = self.dm_ref
        self.dm_total = xp.sum(self.dm_channels, axis=0)

        self.inf_fun_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(self.inf_fun,)))

        xc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2) # DM command coordinates
        yc = self.inf_sampling*(xp.linspace(-self.Nact//2, self.Nact//2-1, self.Nact) + 1/2)

        fx = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf)) # Influence function frequncy sampling
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.Nsurf))

        self.Mx = xp.exp(-1j*2*np.pi*xp.outer(fx,xc)) # forward DM model MFT matrices
        self.My = xp.exp(-1j*2*np.pi*xp.outer(yc,fy))
        self.Mx_back = xp.exp(1j*2*np.pi*xp.outer(xc,fx)) # adjoint DM model MFT matrices
        self.My_back = xp.exp(1j*2*np.pi*xp.outer(fy,yc))

        ### INITIALIZE VORTEX PARAMETERS ###
        self.oversample_vortex = 4.096
        self.N_vortex_lres = int(self.npix*self.oversample_vortex)
        self.vortex_win_diam = 30 # diameter of the window to apply with the vortex model
        self.lres_sampling = 1/self.oversample_vortex # low resolution sampling in lam/D per pixel
        self.lres_win_size = int(self.vortex_win_diam/self.lres_sampling)
        w1d = xp.array(windows.tukey(self.lres_win_size, 1, False))
        self.lres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_lres)
        self.vortex_lres = props.make_vortex_phase_mask(self.N_vortex_lres)

        self.hres_sampling = 0.025 # lam/D per pixel; this value is chosen empirically
        self.N_vortex_hres = int(np.round(self.vortex_win_diam/self.hres_sampling))
        self.hres_win_size = int(self.vortex_win_diam/self.hres_sampling)
        w1d = xp.array(windows.tukey(self.hres_win_size, 1, False))
        self.hres_window = utils.pad_or_crop(xp.outer(w1d, w1d), self.N_vortex_hres)
        self.vortex_hres = props.make_vortex_phase_mask(self.N_vortex_hres)

        y,x = (xp.indices((self.N_vortex_hres, self.N_vortex_hres)) - self.N_vortex_hres//2) * self.hres_sampling
        r = xp.sqrt(x**2 + y**2)
        self.hres_dot_mask = r>=0.15

        # DETECTOR PARAMETERS
        self.CAMLO = None
        self.NCAMLO = 1
        self.camlo_shear = None

    def getattr(self, attr):
        return getattr(self, attr)
    
    def setattr(self, attr, val):
        setattr(self, attr, val)

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wl):
        self._wavelength = wl
        self.camsci_pxscl_lamD = self.camsci_pxscl_lamDc * self.wavelength_c/wl
        self.camlo_pxscl_lamD = self.camlo_pxscl_lamDc * self.wavelength_c/wl
    
    def set_opd(self, opd_array):
        self.PREFPM_OPD = opd_array

    def zero_fsm(self,):
        self.FSM_PTT = np.array([0,0,0])
        self.FSM_OPD = 0*self.PTT_MODES[0]

    def set_fsm(self, ptt):
        self.FSM_PTT = ptt
        # self.FSM_OPD = self.FSM_PTT[0]*self.PTT_MODES[0] + self.FSM_PTT[1]*self.PTT_MODES[1] + self.FSM_PTT[2]*self.PTT_MODES[2]
        
        tip_at_pupil_pv = np.tan(self.FSM_PTT[1]/self.as_per_radian) * self.fsm_beam_diam
        tilt_at_pupil_pv = np.tan(self.FSM_PTT[2]/self.as_per_radian) * self.fsm_beam_diam

        tip_at_pupil_rms = tip_at_pupil_pv * self.tt_pv_to_rms
        tilt_at_pupil_rms = tilt_at_pupil_pv * self.tt_pv_to_rms

        self.FSM_OPD = self.FSM_PTT[0]*self.PTT_MODES[0] + tip_at_pupil_rms*self.PTT_MODES[1] + tilt_at_pupil_rms*self.PTT_MODES[2]

    def add_fsm(self, ptt):
        self.FSM_PTT = self.FSM_PTT + ptt
        # self.FSM_OPD = self.FSM_PTT[0]*self.PTT_MODES[0] + self.FSM_PTT[1]*self.PTT_MODES[1] + self.FSM_PTT[2]*self.PTT_MODES[2]

        tip_at_pupil_pv = np.tan(self.FSM_PTT[1]/self.as_per_radian) * self.fsm_beam_diam
        tilt_at_pupil_pv = np.tan(self.FSM_PTT[2]/self.as_per_radian) * self.fsm_beam_diam

        tip_at_pupil_rms = tip_at_pupil_pv * self.tt_pv_to_rms
        tilt_at_pupil_rms = tilt_at_pupil_pv * self.tt_pv_to_rms

        self.FSM_OPD = self.FSM_PTT[0]*self.PTT_MODES[0] + tip_at_pupil_rms*self.PTT_MODES[1] + tilt_at_pupil_rms*self.PTT_MODES[2]
    
    def get_fsm(self):
        return self.FSM_PTT

    def reset_dm(self):
        self.dm_channels = xp.zeros((10,self.Nact,self.Nact))
        self.dm_channels[0] = self.dm_ref
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def zero_dm(self, channel=1):
        self.dm_channels[channel] = xp.zeros((34,34))
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def set_dm(self, command, channel=1):
        self.dm_channels[channel] = copy.copy(command)
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def add_dm(self, command, channel=1):
        old = self.dm_channels[channel]
        self.dm_channels[channel] = copy.copy(old + command)
        self.dm_total = xp.sum(self.dm_channels, axis=0)

    def get_dm(self, channel=1):
        return copy.copy(self.dm_channels[channel])

    def get_dm_total(self):
        return self.dm_total

    def compute_dm_phasor(self):
        mft_command = self.Mx @ self.dm_total @ self.My
        fourier_surf = self.inf_fun_fft * mft_command
        dm_surf = xp.fft.fftshift( xp.fft.ifft2( xp.fft.ifftshift( fourier_surf, ))).real
        dm_phasor = xp.exp(1j * 4*xp.pi/self.wavelength * dm_surf )
        dm_phasor = utils.pad_or_crop(dm_phasor, self.N)
        return dm_phasor

    def apply_vortex(self, pupwf, plot=False):
        N = pupwf.shape[0]

        lres_wf = utils.pad_or_crop(pupwf, self.N_vortex_lres) # pad to the larger array for the low res propagation
        fp_wf_lres = props.fft(lres_wf)
        fp_wf_lres *= self.vortex_lres * (1 - self.lres_window) # apply low res (windowed) FPM
        pupil_wf_lres = props.ifft(fp_wf_lres)
        pupil_wf_lres = utils.pad_or_crop(pupil_wf_lres, N) # crop to the desired wavefront dimension
        if plot: 
            utils.imshow(
                [xp.abs(pupil_wf_lres), xp.angle(pupil_wf_lres)], 
                titles=['FFT Lyot Pupil Amplitude', 'FFT Lyot Pupil Phase'], 
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )

        fp_wf_hres = props.mft_forward(pupwf, self.npix, self.N_vortex_hres, self.hres_sampling, convention='-')
        fp_wf_hres *= self.vortex_hres * self.hres_window * self.hres_dot_mask # apply high res (windowed) FPM
        pupil_wf_hres = props.mft_reverse(fp_wf_hres, self.hres_sampling, self.npix, N, convention='+')
        if plot: 
            utils.imshow(
                [xp.abs(pupil_wf_hres), xp.angle(pupil_wf_hres)], 
                titles=['MFT Lyot Pupil Amplitude', 'MFT Lyot Pupil Phase'],
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )

        post_vortex_pup_wf = (pupil_wf_lres + pupil_wf_hres)
        if plot: 
            utils.imshow(
                [xp.abs(post_vortex_pup_wf), xp.angle(post_vortex_pup_wf)], 
                titles=['Total Lyot Pupil Amplitude', 'Total Lyot Pupil Phase'],
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )

        return post_vortex_pup_wf

    def calc_wfs_camsci(self, return_all=True, plot=False): # method for getting the PSF in photons
        FSM_PHASOR = utils.pad_or_crop( xp.exp(1j * 4*xp.pi/self.wavelength * self.FSM_OPD ), self.N)
        PREFPM_WFE = utils.pad_or_crop( self.PREFPM_AMP * xp.exp(1j * 2*xp.pi/self.wavelength * self.PREFPM_OPD ), self.N)
        E_EP =  self.APERTURE.astype(complex) * PREFPM_WFE * FSM_PHASOR
        if plot: 
            utils.imshow(
                [xp.abs(E_EP), xp.angle(E_EP)], 
                titles=['EP WF'], 
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )

        DM_PHASOR = self.compute_dm_phasor()
        E_DM = E_EP * DM_PHASOR
        if plot: 
            utils.imshow(
                [xp.abs(E_DM), xp.angle(E_DM)], 
                titles=['After DM WF'], 
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )
        # print(E_DM.shape)

        if self.use_vortex: 
            E_LP = self.apply_vortex(E_DM, plot=plot)
        else: 
            E_LP = copy.copy(E_DM)
        # print(E_LP.shape)

        POSTFPM_WFE = utils.pad_or_crop( self.POSTFPM_AMP * xp.exp(1j * 2*xp.pi/self.wavelength * self.POSTFPM_OPD ) , self.N) + self.SCC_PINHOLE
        E_LP =  E_LP * POSTFPM_WFE
        if plot: 
            utils.imshow(
                [xp.abs(E_LP), xp.angle(E_LP)], 
                titles=['At Lyot Pupil WF'], 
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )

        if self.use_scc:
            E_LS = E_LP * utils.pad_or_crop(self.LYOTSCC, E_LP.shape[0]).astype(complex)
        else: 
            E_LS = E_LP * utils.pad_or_crop(self.LYOTSTOP, E_LP.shape[0]).astype(complex)
        if plot: 
            utils.imshow(xp.abs(E_LS), xp.angle(E_LS), 'After Lyot Stop WF', cmap2='twilight', npix=int(self.plot_oversample*self.npix))

        E_CAMSCI = props.mft_forward(E_LS, self.npix*self.lyot_ratio, self.ncamsci, self.camsci_pxscl_lamD)
        if plot: 
            utils.imshow(
                [xp.abs(E_CAMSCI), xp.angle(E_CAMSCI)], 
                titles=['CAMSCI WF'], 
                norms=[LogNorm(), None], 
                cmaps=['magma', 'twilight'], 
            )

        if return_all:
            return E_EP, DM_PHASOR, E_DM, E_LP, E_LS, E_CAMSCI
        else:
            return E_CAMSCI
    
    def calc_wfs_camlo(self, return_all=True, plot=False): # method for getting the PSF in photons
        FSM_PHASOR = utils.pad_or_crop( xp.exp(1j * 4*xp.pi/self.wavelength * self.FSM_OPD ), self.N)
        PREFPM_WFE = utils.pad_or_crop( self.PREFPM_AMP * xp.exp(1j * 2*xp.pi/self.wavelength * self.PREFPM_OPD ), self.N)
        E_EP =  self.APERTURE.astype(complex) * PREFPM_WFE * FSM_PHASOR
        if plot: 
            utils.imshow(
                [xp.abs(E_EP), xp.angle(E_EP)], 
                titles=['EP WF'], 
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )

        DM_PHASOR = self.compute_dm_phasor()
        E_DM = E_EP * DM_PHASOR
        if plot: 
            utils.imshow(
                [xp.abs(E_DM), xp.angle(E_DM)], 
                titles=['After DM WF'], 
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )
        
        if self.use_vortex: 
            E_DM = utils.pad_or_crop(E_DM, self.Nrls)
            E_LP = self.apply_vortex(E_DM, plot=plot)
        else: 
            E_LP = copy.copy(E_DM)
        # print(E_LP.shape)

        RLS_WFE = self.RLS_AMP * xp.exp(1j * 2*xp.pi/self.wavelength * self.RLS_OPD )
        E_RLS =  E_LP * utils.pad_or_crop(self.RLS, E_LP.shape[0]).astype(complex) * utils.pad_or_crop(RLS_WFE, E_LP.shape[0])
        if plot: 
            utils.imshow(
                [xp.abs(E_RLS), xp.angle(E_RLS)], 
                titles=['At RLS WF'], 
                npix=2*[int(self.plot_oversample*self.npix)], 
                cmaps=['plasma', 'twilight'], 
            )

        # Use TF and MFT to propagate to defocused image
        self.llowfsc_fnum = self.camlo_fl/self.lyot_diam
        camlo_defocus_tf = props.get_fresnel_TF(
            self.llowfsc_defocus * self.rls_oversample**2, 
            self.Nrls, 
            self.wavelength, 
            self.llowfsc_fnum,
        )
        E_CAMLO = props.mft_forward(camlo_defocus_tf*E_RLS, self.npix*self.lyot_ratio, self.ncamlo, self.camlo_pxscl_lamD)
        if self.camlo_shear is not None: # shift the CAMLO image to simulate detector lateral shift
            E_CAMLO = xcipy.ndimage.shift(E_CAMLO, (self.camlo_shear[1], self.camlo_shear[0]), order=3)

        if plot: 
            utils.imshow(
                [xp.abs(E_CAMLO), xp.angle(E_CAMLO)], 
                titles=['CAMLO WF'], 
                cmaps=['magma', 'twilight'], 
            )
            
        if return_all:
            return E_EP, DM_PHASOR, E_DM, E_LP, E_RLS, E_CAMLO
        else:
            return E_CAMLO
    
    def calc_wf_camsci(self):
        fpwf = self.calc_wfs_camsci( return_all=False ) / xp.sqrt(self.Imax_ref)
        return fpwf
    
    def snap_camsci(self):
        image = xp.abs(self.calc_wfs_camsci(return_all=False))**2 / self.Imax_ref
        return image
    
    def snap_camlo(self):
        camlo_im = xp.abs(self.calc_wfs_camlo(return_all=False))**2
        if self.CAMLO is not None:
            noisy_im = 0.0
            for i in range(self.NCAMLO):
                noisy_im += self.CAMLO.add_noise(camlo_im)
            return noisy_im/self.NCAMLO
        return camlo_im

class parallel():
    def __init__(
            self,
            ACTORS,
        ):

        self.ACTORS = ACTORS
        self.Nactors = len(ACTORS)

        self.wavelength_c = self.getattr('wavelength_c')
        self.total_pupil_diam = self.getattr('total_pupil_diam')
        self.fsm_beam_diam = self.getattr('fsm_beam_diam')
        self.dm_beam_diam = self.getattr('dm_beam_diam')
        self.lyot_pupil_diam = self.getattr('lyot_pupil_diam')
        self.lyot_diam = self.getattr('lyot_diam')
        self.lyot_ratio = self.getattr('lyot_ratio')
        self.rls_diam = self.getattr('rls_diam')
        self.camsci_fl = self.getattr('camsci_fl')
        self.llowfsc_fl = self.getattr('llowfsc_fl')
        self.llowfsc_fnum  = self.getattr('llowfsc_fnum')
        self.llowfsc_defocus = self.getattr('llowfsc_defocus')
        self.camsci_pxscl = self.getattr('camsci_pxscl')
        self.camsci_pxscl_lamDc = self.getattr('camsci_pxscl_lamDc')
        self.camlo_pxscl = self.getattr('camlo_pxscl_lamDc')
        self.camlo_pxscl_lamDc = self.getattr('camlo_pxscl_lamDc')

        self.ncamsci = self.getattr('ncamsci')
        self.ncamlo = self.getattr('ncamlo')

        self.APERTURE = ray.get(ACTORS[0].getattr.remote('APERTURE'))
        self.PTT_MODES = ray.get(ACTORS[0].getattr.remote('PTT_MODES'))

        self.Nact = ray.get(ACTORS[0].getattr.remote('Nact'))
        self.dm_mask = ray.get(ACTORS[0].getattr.remote('dm_mask'))
        self.dm_ref = ray.get(ACTORS[0].getattr.remote('dm_ref'))
        self.reset_dm()

        # DETECTOR PARAMETERS
        self.CAMSCI = None
        self.NCAMSCI = 1
        self.Imax_ref = 1

        self.CAMLO = None
        self.NCAMLO = 1

    def getattr(self, attr):
        return ray.get(self.ACTORS[0].getattr.remote(attr))
    
    def set_actor_attr(self, attr, value):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].setattr.remote(attr, value)
    
    def set_opd(self, opd_array):
        # self.PREFPM_OPD = opd_array
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].set_opd.remote(opd_array)

    def zero_fsm(self,):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].zero_fsm.remote()

    def set_fsm(self, ptt):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].set_fsm.remote( ptt )
        
    def add_fsm(self, ptt):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].add_fsm.remote( ptt )

    def get_fsm(self):
        return self.getattr('FSM_PTT')

    def reset_dm(self):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].reset_dm.remote()

    def zero_dm(self, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].zero_dm.remote(channel)

    def set_dm(self, command, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].set_dm.remote(command, channel)

    def add_dm(self, command, channel=1):
        for i in range(len(self.ACTORS)):
            self.ACTORS[i].add_dm.remote(command, channel)

    def get_dm(self, channel=1):
        return copy.copy(self.getattr('dm_channels')[channel])

    def get_dm_total(self):
        return self.getattr('dm_total')
    
    def snap_camsci(self):
        pending_ims = []
        for i in range(self.Nactors):
            future_ims = self.ACTORS[i].snap_camsci.remote()
            pending_ims.append(future_ims)

        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        camsci_im = xp.sum(ims, axis=0)/self.Imax_ref

        return camsci_im

    def snap_camlo(self):
        pending_ims = []
        for i in range(self.Nactors):
            future_ims = self.ACTORS[i].snap_camlo.remote()
            pending_ims.append(future_ims)
            
        ims = ray.get(pending_ims)
        ims = xp.array(ims)
        camlo_im = xp.sum(ims, axis=0)

        if self.CAMLO is not None:
            noisy_im = 0.0
            for i in range(self.NCAMLO):
                noisy_im += self.CAMLO.add_noise(camlo_im)

            camlo_im = noisy_im / self.NCAMLO

        return camlo_im