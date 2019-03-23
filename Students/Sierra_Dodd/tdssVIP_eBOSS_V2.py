from pylab import *
import urllib
import pyfits
import numpy as np
from scipy.signal import boxcar
import sys
from scipy.stats import scoreatpercentile

##########################################################################
DEG_PER_HR = 360. / 24.             # degrees per hour
DEG_PER_MIN = DEG_PER_HR / 60.      # degrees per min
DEG_PER_S = DEG_PER_MIN / 60.       # degrees per sec
DEG_PER_AMIN = 1./60.               # degrees per arcmin
DEG_PER_ASEC = DEG_PER_AMIN / 60.   # degrees per arcsec
RAD_PER_DEG = pi / 180.             # radians per degree
def ang_sep(ra1, dec1, ra2, dec2):
    ra1 = np.asarray(ra1);  ra2 = np.asarray(ra2)
    dec1 = np.asarray(dec1);  dec2 = np.asarray(dec2)

    ra1 = ra1 * RAD_PER_DEG           # convert to radians
    ra2 = ra2 * RAD_PER_DEG
    dec1 = dec1 * RAD_PER_DEG
    dec2 = dec2 * RAD_PER_DEG

    sra1 = sin(ra1);  sra2 = sin(ra2)
    cra1 = cos(ra1);  cra2 = cos(ra2)
    sdec1 = sin(dec1);  sdec2 = sin(dec2)
    cdec1 = cos(dec1);  cdec2 = cos(dec2)

    csep = cdec1*cdec2*(cra1*cra2 + sra1*sra2) + sdec1*sdec2

    # An ugly work-around for floating point issues.
    #if np.any(csep > 1):  print csep
    csep = np.where(csep > 1., 1., csep)

    degsep = arccos(csep) / RAD_PER_DEG
    # only works for separations > 0.1 of an arcsec or  >~2.7e-5 dec
    degsep = np.where(degsep < 1e-5, 0, degsep)
    return degsep
##########################################################################
def smooth(x):
    window_len=11
    s=r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    w=boxcar(11)
    y=convolve(w/w.sum(),s,mode='valid')
    return y[5:-5]
##########################################################################
def get_ebosstarget_class(target_bit0, target_bit1, target_bit2):
    # use eboss_target2 flag to get subclasses
    target_classes = []
    if (target_bit0 & 2**10) != 0: target_classes.append('SEQUELS_QSO_EBOSS_CORE')
    if (target_bit0 & 2**11) != 0: target_classes.append('SEQUELS_QSO_PTF')
    if (target_bit0 & 2**12) != 0: target_classes.append('SEQUELS_QSO_REOBS')
    if (target_bit0 & 2**13) != 0: target_classes.append('SEQUELS_QSO_EBOSS_KDE')
    if (target_bit0 & 2**14) != 0: target_classes.append('SEQUELS_QSO_EBOSS_FIRST')
    if (target_bit0 & 2**15) != 0: target_classes.append('SEQUELS_QSO_BAD_BOSS')
    if (target_bit0 & 2**16) != 0: target_classes.append('SEQUELS_QSO_QSO_BOSS_TARGET')
    if (target_bit0 & 2**17) != 0: target_classes.append('SEQUELS_QSO_SDSS_TARGET')
    if (target_bit0 & 2**18) != 0: target_classes.append('SEQUELS_QSO_KNOWN')
    if (target_bit0 & 2**19) != 0: target_classes.append('SEQUELS_DR9_CALIB_TARGET')
    if (target_bit0 & 2**20) != 0: target_classes.append('SEQUELS_SPIDERS_RASS_AGN')
    if (target_bit0 & 2**21) != 0: target_classes.append('SEQUELS_SPIDERS_RASS_CLUS')
    if (target_bit0 & 2**22) != 0: target_classes.append('SEQUELS_SPIDERS_ERASS_AGN')
    if (target_bit0 & 2**23) != 0: target_classes.append('SEQUELS_SPIDERS_ERASS_CLUS')
    if (target_bit0 & 2**30) != 0: target_classes.append('SEQUELS_TDSS_A')
    if (target_bit0 & 2**31) != 0: target_classes.append('SEQUELS_TDSS_FES_DE')
    if (target_bit0 & 2**32) != 0: target_classes.append('SEQUELS_TDSS_FES_DWARFC')
    if (target_bit0 & 2**33) != 0: target_classes.append('SEQUELS_TDSS_FES_NQHISN')
    if (target_bit0 & 2**34) != 0: target_classes.append('SEQUELS_TDSS_FES_MGII')
    if (target_bit0 & 2**35) != 0: target_classes.append('SEQUELS_TDSS_FES_VARBAL')
    if (target_bit0 & 2**40) != 0: target_classes.append('SEQUELS_PTF_VARIABLE')

    if (target_bit1 & 2**9) != 0: target_classes.append('eBOSS_QSO1_VAR_S82')
    if (target_bit1 & 2**10) != 0: target_classes.append('eBOSS_QSO1_EBOSS_CORE')
    if (target_bit1 & 2**11) != 0: target_classes.append('eBOSS_QSO1_PTF')
    if (target_bit1 & 2**12) != 0: target_classes.append('eBOSS_QSO1_REOBS')
    if (target_bit1 & 2**13) != 0: target_classes.append('eBOSS_QSO1_EBOSS_KDE')
    if (target_bit1 & 2**14) != 0: target_classes.append('eBOSS_QSO1_EBOSS_FIRST')
    if (target_bit1 & 2**15) != 0: target_classes.append('eBOSS_QSO1_BAD_BOSS')
    if (target_bit1 & 2**16) != 0: target_classes.append('eBOSS_QSO_BOSS_TARGET')
    if (target_bit1 & 2**17) != 0: target_classes.append('eBOSS_QSO_SDSS_TARGET')
    if (target_bit1 & 2**18) != 0: target_classes.append('eBOSS_QSO_KNOWN')
    if (target_bit1 & 2**30) != 0: target_classes.append('TDSS_TARGET')
    if (target_bit1 & 2**31) != 0: target_classes.append('SPIDERS_TARGET')

    if (target_bit2 & 2**0) != 0: target_classes.append('SPIDERS_RASS_AGN')
    if (target_bit2 & 2**1) != 0: target_classes.append('SPIDERS_RASS_CLUS')
    if (target_bit2 & 2**2) != 0: target_classes.append('SPIDERS_ERASS_AGN')
    if (target_bit2 & 2**3) != 0: target_classes.append('SPIDERS_ERASS_CLUS')
    if (target_bit2 & 2**4) != 0: target_classes.append('SPIDERS_XMMSL_AGN')
    if (target_bit2 & 2**5) != 0: target_classes.append('SPIDERS_XCLASS_CLUS')
    if (target_bit2 & 2**5) != 0: target_classes.append('SPIDERS_XCLASS_CLUS')
    if (target_bit2 & 2**20) != 0: target_classes.append('TDSS_A')
    if (target_bit2 & 2**21) != 0: target_classes.append('TDSS_FES_DE')
    if (target_bit2 & 2**22) != 0: target_classes.append('TDSS_FES_DWARFC')
    if (target_bit2 & 2**23) != 0: target_classes.append('TDSS_FES_NQHISN')
    if (target_bit2 & 2**24) != 0: target_classes.append('TDSS_FES_MGII')
    if (target_bit2 & 2**25) != 0: target_classes.append('TDSS_FES_VARBAL')
    if (target_bit2 & 2**26) != 0: target_classes.append('TDSS_B')
    if (target_bit2 & 2**27) != 0: target_classes.append('TDSS_FES_HYPQSO')
    if (target_bit2 & 2**28) != 0: target_classes.append('TDSS_FES_HYPSTAR')
    if (target_bit2 & 2**29) != 0: target_classes.append('TDSS_FES_WDDM')
    if (target_bit2 & 2**30) != 0: target_classes.append('TDSS_FES_ACTSTAR')
    if (target_bit2 & 2**31) != 0: target_classes.append('TDSS_COREPTF')

    return target_classes

##########################################################################

class platespectra:
    def __init__(self, current_platefile, current_spZbestfile):
        npix = len(current_platefile[0].data[0])
        coeff1 = current_platefile[0].header['COEFF1']
        coeff0 = current_platefile[0].header['COEFF0']
        self.plate = current_platefile[0].header['PLATEID']
        self.mjd = current_platefile[0].header['MJD']
        self.wavelength = 10.**(coeff0+coeff1*arange(npix))
        self.spectra = current_platefile[0].data
        self.ivar = current_platefile[1].data
        self.fluxerr = 1./sqrt(self.ivar)
        self.redshift = current_spZbestfile[1].data['Z']
        self.plug_ra = current_spZbestfile[1].data['PLUG_RA']
        self.plug_dec = current_spZbestfile[1].data['PLUG_DEC']
        self.class0 = current_spZbestfile[1].data['CLASS']
        self.subclass0 = current_spZbestfile[1].data['SUBCLASS']
        self.fiberid = current_spZbestfile[1].data['FIBERID']
        self.source_type = current_platefile[5].data['SOURCETYPE']
        self.obj_type = current_platefile[5].data['OBJTYPE']
        self.boss_target2 = current_platefile[5].data['BOSS_TARGET2']
        self.boss_target1 = current_platefile[5].data['BOSS_TARGET1']
        self.ancillary_target2 = current_platefile[5].data['ANCILLARY_TARGET2']
        self.ancillary_target1 = current_platefile[5].data['ANCILLARY_TARGET1']
        self.eboss_target0 = current_platefile[5].data['EBOSS_TARGET0']
        self.eboss_target1 = current_platefile[5].data['EBOSS_TARGET1']
        self.eboss_target2 = current_platefile[5].data['EBOSS_TARGET2']
        self.sourcetype = current_platefile[5].data['SOURCETYPE']
        self.model = current_spZbestfile[2].data

##########################################################################

class PS1_3pi_lightcurves:
    def __init__(self, maxra, minra, maxdec, mindec):
        self.fitsfile = ()
        self.fitsfile += ('TDSS_candidates_chunk_1_out.fits',)
        self.fitsfile += ('TDSS_candidates_chunk_1_ptf_core.fits',)
        self.fitsfile += ('TDSS_candidates_chunk_2_out.fits',)
        self.fitsfile += ('TDSS_candidates_chunk_2_ptf_core.fits',)
        self.fitsfile += ('TDSS_candidates_chunk_3_out.fits',)
        self.fitsfile += ('TDSS_candidates_chunk_3_ptf_core.fits',)
        self.fitsfile += ('TDSS_candidates_chunk_4_out.fits',)
        self.fitsfile += ('TDSS_candidates_chunk_4_out_extra.fits',)
        self.fitsfile += ('TDSS_candidates_chunk_4_ptf_core.fits',)
        self.PS1_ra = []; self.PS1_dec = []; self.PS1_obj_id = []
        self.PS1_median = []; self.PS1_err = []
        self.PS1_sdss_ps1 = []; self.PS1_sdss_ps1_err = []; self.PS1_sdss_mjd = []
        self.PS1_lc_mag_g = []; self.PS1_lc_err_g = []; self.PS1_lc_mjd_g = []
        self.PS1_lc_mag_r = []; self.PS1_lc_err_r = []; self.PS1_lc_mjd_r = []
        self.PS1_lc_mag_i = []; self.PS1_lc_err_i = []; self.PS1_lc_mjd_i = []

        for filename in self.fitsfile:
#            PS1file = urllib.urlretrieve('http://students.washington.edu/jruan//PS3pi_lightcurves/eBOSS/'+filename)
            PS1file = urllib.urlretrieve('http://faculty.washington.edu/sfander/tdssVIP/PS3pi_lightcurves/eBOSS/'+filename)
            PS1file = pyfits.open(PS1file[0])
            self.PS1_ra += list(PS1file[1].data['ra'])
            self.PS1_dec += list(PS1file[1].data['dec'])
            self.PS1_median += list(PS1file[1].data['median'])
            self.PS1_err += list(PS1file[1].data['err'])
            self.PS1_sdss_ps1 += list(PS1file[1].data['sdss_ps1'])
            self.PS1_sdss_ps1_err += list(PS1file[1].data['sdss_ps1_err'])
            self.PS1_sdss_mjd += list(PS1file[1].data['sdss_mjd'])
            self.PS1_lc_mag_g += list(PS1file[1].data['lc_mag_g'])
            self.PS1_lc_err_g += list(PS1file[1].data['lc_err_g'])
            self.PS1_lc_mjd_g += list(PS1file[1].data['lc_mjd_g'])
            self.PS1_lc_mag_r += list(PS1file[1].data['lc_mag_r'])
            self.PS1_lc_err_r += list(PS1file[1].data['lc_err_r'])
            self.PS1_lc_mjd_r += list(PS1file[1].data['lc_mjd_r'])
            self.PS1_lc_mag_i += list(PS1file[1].data['lc_mag_i'])
            self.PS1_lc_err_i += list(PS1file[1].data['lc_err_i'])
            self.PS1_lc_mjd_i += list(PS1file[1].data['lc_mjd_i'])

    def find_PS1_counterpart(self, fiber_ra, fiber_dec):
        ang_separations = ang_sep(fiber_ra, fiber_dec, asarray(self.PS1_ra), asarray(self.PS1_dec))
        matched_index = where(ang_separations == min(ang_separations))[0][0]
        matched_dist = min(ang_separations)*3600.
        return matched_index, matched_dist

    def get_3pi_lightcurve(self):
        global PS1_matched_index
        g_mag = self.PS1_lc_mag_g[PS1_matched_index][where(self.PS1_lc_mag_g[PS1_matched_index] > 10)]
        r_mag = self.PS1_lc_mag_r[PS1_matched_index][where(self.PS1_lc_mag_r[PS1_matched_index] > 10)]
        i_mag = self.PS1_lc_mag_i[PS1_matched_index][where(self.PS1_lc_mag_i[PS1_matched_index] > 10)]
        g_mjd = self.PS1_lc_mjd_g[PS1_matched_index][where(self.PS1_lc_mag_g[PS1_matched_index] > 10)]
        r_mjd = self.PS1_lc_mjd_r[PS1_matched_index][where(self.PS1_lc_mag_r[PS1_matched_index] > 10)]
        i_mjd = self.PS1_lc_mjd_i[PS1_matched_index][where(self.PS1_lc_mag_i[PS1_matched_index] > 10)]
        g_err = self.PS1_lc_err_g[PS1_matched_index][where(self.PS1_lc_mag_g[PS1_matched_index] > 10)]
        r_err = self.PS1_lc_err_r[PS1_matched_index][where(self.PS1_lc_mag_r[PS1_matched_index] > 10)]
        i_err = self.PS1_lc_err_i[PS1_matched_index][where(self.PS1_lc_mag_i[PS1_matched_index] > 10)]
        return g_mag, r_mag, i_mag, g_mjd, r_mjd, i_mjd, g_err, r_err, i_err

    def ug_gr_contours(self):
        x = [m[0]-m[1] for m in self.PS1_median if (-1.<(m[0]-m[1])<2.) and (-1.<(m[1]-m[2])<2.)]
        y = [m[1]-m[2] for m in self.PS1_median if (-1.<(m[1]-m[2])<2.) and (-1.<(m[0]-m[1])<2.)]
        npts = len(x)
        xmin = float(min(x)); xmax = float(max(x))
        ymin = float(min(y)); ymax = float(max(y))
        dx = (xmax - xmin) /80; dy = (ymax - ymin) /80
        x_range = np.arange(xmin, xmax, dx)
        y_range = np.arange(ymin, ymax, dy)
        nbinsx = x_range.shape[0]; nbinsy = y_range.shape[0]        
        Z = np.zeros((nbinsy, nbinsx))
        xbin = np.zeros(npts); ybin = np.zeros(npts)
        # assign each point to a bin
        for i in range(npts):
            xbin[i] = min(int((x[i] - xmin) / dx), nbinsx-1)
            ybin[i] = min(int((y[i] - ymin) / dy), nbinsy-1)
            # and count how many are in each bin
#            Z[ ybin[i] ][ xbin[i] ] += 1
            Z[ int(ybin[i]) ][ int(xbin[i]) ] += 1
        return Z, [xmin, xmax, ymin, ymax]

###############################################################
# callback handling of buttons
class callback_handler:
    def __init__(self):
        self.specframe = 0 # 0 = obsframe, 1 = restframe

    def next(self):
        global ax1, ax2, ax3, ax4
        global TDSS_fiber_index, PS1_matched_index, PS1_matched_dist
        delaxes(ax1); delaxes(ax2); delaxes(ax3); delaxes(ax4)
        self.specframe = 0

        build_title = 'plate = '+str(currentplate.plate)+', mjd = '+str(currentplate.mjd)+', sourcetype = '+currentplate.source_type[TDSS_fiber_index]+', objtype = '+currentplate.obj_type[TDSS_fiber_index]+'\n Targeted by: '
        for target_class in get_ebosstarget_class(currentplate.eboss_target0[TDSS_fiber_index], currentplate.eboss_target1[TDSS_fiber_index], currentplate.eboss_target2[TDSS_fiber_index]):
            build_title += target_class+', '
        title.set_text(build_title)

        ax1 = subplot2grid((2,3), (0,0),colspan=1)
        SDSS_cutout=imread(urllib.urlretrieve('http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra='+str(currentplate.plug_ra[TDSS_fiber_index])+'&dec='+str(currentplate.plug_dec[TDSS_fiber_index])+'&scale=0.15&width=400&height=400&opt=GL')[0])
        ax1.implot = imshow(SDSS_cutout)
        ax1.xaxis.set_visible(False); ax1.yaxis.set_visible(False)
        ax1.set_title('ra = '+str(currentplate.plug_ra[TDSS_fiber_index])+', dec = '+str(currentplate.plug_dec[TDSS_fiber_index]), fontsize=14)
        
        ax2 = subplot2grid((2,3), (0,1),colspan=2)
        ticklabels = ax2.get_xticklabels()
        ticklabels.extend( ax2.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(12)
        ax2.set_ylabel(r'F$_\lambda$ [10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', size=12)
        ax2.set_xlabel(r'wavelength [$\AA$]', size=12)
        ax2.set_title('fiber = '+str(TDSS_fiber_index+1)+', class = '+currentplate.class0[TDSS_fiber_index]+', subclass = '+currentplate.subclass0[TDSS_fiber_index]+', z = '+str(round(currentplate.redshift[TDSS_fiber_index], 3)), fontsize=14)
        ax2.errorbar(currentplate.wavelength, smooth(currentplate.spectra[TDSS_fiber_index]), yerr=smooth(currentplate.fluxerr[TDSS_fiber_index]), ecolor='paleturquoise', color='k', capsize=0.)
        ax2.plot(currentplate.wavelength, currentplate.model[TDSS_fiber_index], 'r-')
	ax3 = subplot2grid((2,3), (1,0),colspan=1)
	ax4 = subplot2grid((2,3), (1,1),colspan=2)
        if(PS1_matched_dist < 10):
	    ax3.set_title('matched distance = '+str(round(PS1_matched_dist,2))+'"')
            ax3.contour(gr_ri_density, extent=contour_extent)
            ticklabels = ax3.get_xticklabels()
            ticklabels.extend( ax3.get_yticklabels() )
            for label in ticklabels:
                label.set_fontsize(12)
            ax3.set_ylabel(r'$g - r$', size=14)
            ax3.set_xlabel(r'$u - g$', size=14)
            ax3.plot(PS1_lc.PS1_sdss_ps1[PS1_matched_index][0] - PS1_lc.PS1_sdss_ps1[PS1_matched_index][1], PS1_lc.PS1_sdss_ps1[PS1_matched_index][1] - PS1_lc.PS1_sdss_ps1[PS1_matched_index][2], 'r*', markersize=12)

            g_mag, r_mag, i_mag, g_mjd, r_mjd, i_mjd, g_err, r_err, i_err = PS1_lc.get_3pi_lightcurve() # get lightcurves of current PS1_matched_index
            ticklabels = ax4.get_xticklabels()
            ticklabels.extend( ax4.get_yticklabels() )
            for label in ticklabels:
                ax4.set_ylabel(r'PS1 mag', size=12)
                ax4.set_xlabel(r'MJD', size=12)
                ax4 = gca()
                ax4.invert_yaxis()
            maxx = 0; minx = 999999.
            if len(g_mag)>0: 
                ax4.errorbar(g_mjd, g_mag, yerr = g_err, ecolor='g', color='g', fmt='o', label=r'$g$')
                if np.min(g_mjd)<minx: minx = np.min(g_mjd)
                if np.max(g_mjd)>maxx: maxx = np.max(g_mjd)
            if len(r_mag)>0:	
                ax4.errorbar(r_mjd, r_mag, yerr = r_err, ecolor='r', color='r', fmt='o', label=r'$r$')
                if np.min(r_mjd)<minx: minx = np.min(r_mjd)
                if np.max(r_mjd)>maxx: maxx = np.max(r_mjd)
            if len(i_mag)>0: 
                ax4.errorbar(i_mjd, i_mag, yerr = i_err, ecolor='b', color='b', fmt='o', label=r'$i$')
                if np.min(i_mjd)<minx: minx = np.min(i_mjd)
                if np.max(i_mjd)>maxx: maxx = np.max(i_mjd)
            ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][0],PS1_lc.PS1_sdss_ps1[PS1_matched_index][0]], 'g--')
            ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][1],PS1_lc.PS1_sdss_ps1[PS1_matched_index][1]], 'r--')
            ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][2],PS1_lc.PS1_sdss_ps1[PS1_matched_index][2]], 'b--')
            ax4.legend(loc=3, prop={'size':12}, ncol=3, numpoints=1, columnspacing = 0.2, handletextpad=.01, borderpad=.02)
            draw()

    def accept(self, event):
        global inspection_flags
        global TDSS_fiber_indicies, TDSS_fiber_index
        global PS1_matched_indicies, PS1_matched_index
        global PS1_matched_dists, PS1_matched_dist

        current_index = where(TDSS_fiber_indicies==TDSS_fiber_index)[0][0]
        inspection_flags[current_index] = 1
        
        next_index = current_index+1
        if next_index == len(TDSS_fiber_indicies):
            self.write2file()
        else:
            TDSS_fiber_index = TDSS_fiber_indicies[next_index]
            PS1_matched_index = PS1_matched_indicies[next_index]
            PS1_matched_dist = PS1_matched_dists[next_index]
            next(self)

    def flag(self, event):
        global inspection_flags
        global TDSS_fiber_indicies, TDSS_fiber_index
        global PS1_matched_indicies, PS1_matched_index
        global PS1_matched_dists, PS1_matched_dist

        current_index = where(TDSS_fiber_indicies==TDSS_fiber_index)[0][0]
        inspection_flags[current_index] = 2
        
        next_index = current_index+1
        if next_index == len(TDSS_fiber_indicies):
            self.write2file()
        else:
            TDSS_fiber_index = TDSS_fiber_indicies[next_index]
            PS1_matched_index = PS1_matched_indicies[next_index]
            PS1_matched_dist = PS1_matched_dists[next_index]
            next(self)

    def back(self, event):
        global inspection_flags
        global TDSS_fiber_indicies, TDSS_fiber_index
        global PS1_matched_indicies, PS1_matched_index
        global PS1_matched_dists, PS1_matched_dist

        current_index = where(TDSS_fiber_indicies==TDSS_fiber_index)[0][0]
        next_index = current_index+-1
        TDSS_fiber_index = TDSS_fiber_indicies[next_index]
        PS1_matched_index = PS1_matched_indicies[next_index]
        PS1_matched_dist = PS1_matched_dists[next_index]
        next(self)

    def goforward(self, event):
        global TDSS_fiber_indicies, TDSS_fiber_index
        global PS1_matched_indicies, PS1_matched_index
        global PS1_matched_dists, PS1_matched_dist

        current_index = where(TDSS_fiber_indicies==TDSS_fiber_index)[0][0]
        
        next_index = current_index+1
        if next_index == len(TDSS_fiber_indicies):
            close()
        else:
            TDSS_fiber_index = TDSS_fiber_indicies[next_index]
            PS1_matched_index = PS1_matched_indicies[next_index]
            PS1_matched_dist = PS1_matched_dists[next_index]
            next(self)

    def goback(self, event):
        global TDSS_fiber_indicies, TDSS_fiber_index
        global PS1_matched_indicies, PS1_matched_index
        global PS1_matched_dists, PS1_matched_dist

        current_index = where(TDSS_fiber_indicies==TDSS_fiber_index)[0][0]
        
        next_index = current_index-1
        TDSS_fiber_index = TDSS_fiber_indicies[next_index]
        PS1_matched_index = PS1_matched_indicies[next_index]
        PS1_matched_dist = PS1_matched_dists[next_index]
        next(self)

    def restframe(self, event):
        global TDSS_fiber_index, ax2
        if (self.specframe == 0):
            delaxes(ax2)
            ax2 = subplot2grid((2,3), (0,1),colspan=2)
            ticklabels = ax2.get_xticklabels()
            ticklabels.extend(ax2.get_yticklabels())
            for label in ticklabels:
                label.set_fontsize(12)
            ax2.set_ylabel(r'F$_\lambda$ [10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', size=12)
            ax2.set_xlabel(r'wavelength [$\AA$]', size=12)
            ax2.set_title('fiber = '+str(TDSS_fiber_index+1)+', class = '+currentplate.class0[TDSS_fiber_index]+', subclass = '+currentplate.subclass0[TDSS_fiber_index]+', z = '+str(round(currentplate.redshift[TDSS_fiber_index], 3)), fontsize=14)
            ax2.errorbar(currentplate.wavelength/(1.+currentplate.redshift[TDSS_fiber_index]), smooth(currentplate.spectra[TDSS_fiber_index]), yerr=smooth(currentplate.fluxerr[TDSS_fiber_index]), ecolor='paleturquoise', color='k', capsize=0.)
            ax2.plot(currentplate.wavelength/(1.+currentplate.redshift[TDSS_fiber_index]), currentplate.model[TDSS_fiber_index], 'r-')
        self.specframe = 1

    def obsframe(self, event):
        global TDSS_fiber_index, ax2
        if (self.specframe == 1):
            delaxes(ax2)
            ax2 = subplot2grid((2,3), (0,1),colspan=2)
            ticklabels = ax2.get_xticklabels()
            ticklabels.extend(ax2.get_yticklabels())
            for label in ticklabels:
                label.set_fontsize(12)
            ax2.set_ylabel(r'F$_\lambda$ [10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', size=12)
            ax2.set_xlabel(r'wavelength [$\AA$]', size=12)
            ax2.set_title('fiber = '+str(TDSS_fiber_index+1)+', class='+currentplate.class0[TDSS_fiber_index]+', z='+str(round(currentplate.redshift[TDSS_fiber_index], 3)), fontsize=14)
            ax2.errorbar(currentplate.wavelength, smooth(currentplate.spectra[TDSS_fiber_index]), yerr=smooth(currentplate.fluxerr[TDSS_fiber_index]), ecolor='paleturquoise', color='k', capsize=0.)
            ax2.plot(currentplate.wavelength, currentplate.model[TDSS_fiber_index], 'r-')
        self.specframe = 0

    def comment(self, event):
        global TDSS_fiber_index, comments
        current_index = where(TDSS_fiber_indicies==TDSS_fiber_index)[0][0]

        comments[current_index] = raw_input('enter comment for fiber '+str(TDSS_fiber_index+1)+': ')
        if len(comments[current_index]) == 0:
            comments[current_index] = 'nc'

    def write2file(self):
	# Updated 3/22/19 to include redux version in filename and add plate and mjd as columns in output file
        close()
        global inspection_flags, TDSS_fiber_indicies
        outputfile = open('tdssVIP_plateresults-'+str(currentplate.plate)+'-'+str(currentplate.mjd)+'-v5_10_10'+'.txt', 'w')
        print 'writing inspection results to file '+'tdssvip_plateresults-'+str(currentplate.plate)+'-'+str(currentplate.mjd)+'-v5_10_10'+'.txt'+' ...'
        outputfile.write('plate \t mjd \t fiber \t ra \t dec \t z \t eboss_target0_bit \t eboss_target1_bit \t eboss_target2_bit \t class_pipeline \t class_VI \t subclass_pipeline \t comment \n')
        for i in range(0, len(inspection_flags)):
            outputfile.write(str(plate)+'\t'+str(mjd)+'\t'+str(TDSS_fiber_indicies[i]+1)+'\t'+str(currentplate.plug_ra[TDSS_fiber_indicies[i]])+'\t'+str(currentplate.plug_dec[TDSS_fiber_indicies[i]])+'\t'+str(currentplate.redshift[TDSS_fiber_indicies[i]])+' \t'+str(currentplate.eboss_target0[TDSS_fiber_indicies[i]])+'\t'+str(currentplate.eboss_target1[TDSS_fiber_indicies[i]])+'\t'+str(currentplate.eboss_target2[TDSS_fiber_indicies[i]])+'\t'+str(currentplate.class0[TDSS_fiber_indicies[i]])+'\t')
            if inspection_flags[i] == 1:
                outputfile.write(str(currentplate.class0[TDSS_fiber_indicies[i]])+'\t')
            elif inspection_flags[i] == 2:
                outputfile.write('FLAGGED'+'\t')
            if len(str(currentplate.subclass0[TDSS_fiber_indicies[i]])) > 0:
                outputfile.write(str(currentplate.subclass0[TDSS_fiber_indicies[i]].replace(' ',''))+'\t')
            else:
                outputfile.write('None\t')
            outputfile.write(comments[i]+'\n')
        outputfile.close()
        print 'done writing!'
###############################################################

if len(sys.argv) == 3:
    plate = str(sys.argv[1]); mjd = str(sys.argv[2])
    platefile = urllib.urlretrieve('http://data.sdss.org/sas/ebosswork/eboss/spectro/redux/v5_10_10/'+plate+'/spPlate-'+plate+'-'+mjd+'.fits')
    platefile = pyfits.open(platefile[0])
    spZbestfile = urllib.urlretrieve('http://data.sdss.org/sas/ebosswork/eboss/spectro/redux/v5_10_10/'+plate+'/v5_10_10/spZbest-'+plate+'-'+mjd+'.fits')
    spZbestfile = pyfits.open(spZbestfile[0])
    currentplate = platespectra(platefile, spZbestfile)
    PS1_lc = PS1_3pi_lightcurves(max(currentplate.plug_ra), min(currentplate.plug_ra), max(currentplate.plug_dec), min(currentplate.plug_dec))
    gr_ri_density, contour_extent = PS1_lc.ug_gr_contours()

    #find plate indices (0-999) of the TDSS targets by matching to PS1 targets file
    ##global PS1_matched_indicies, PS1_matched_dists, TDSS_fiber_indicies #lists 
    PS1_matched_indicies = []; PS1_matched_dists = []; TDSS_fiber_indicies = []
    for current_fiber in range(0, 1000):
        if ((currentplate.eboss_target1[current_fiber] & 2**30) != 0): #if TDSS target
            current_PS1_matched_index, current_PS1_matched_dist = PS1_lc.find_PS1_counterpart(currentplate.plug_ra[current_fiber], currentplate.plug_dec[current_fiber])
            TDSS_fiber_indicies.append(current_fiber)
            PS1_matched_dists.append(current_PS1_matched_dist)
            PS1_matched_indicies.append(current_PS1_matched_index)
    TDSS_fiber_indicies = asarray(TDSS_fiber_indicies); PS1_matched_dists = asarray(PS1_matched_dists); PS1_matched_indicies = asarray(PS1_matched_indicies)

    # set plotting canvas
    figure(figsize=(13, 9), dpi=80)
    subplots_adjust(bottom=.13, wspace = .4, hspace=.25, left=.07, right=.95, top=.88)
    ##global title, inspection_flags #flags from visual inspection, 1=accept, 2=flag
    ##global TDSS_fiber_index; 
    TDSS_fiber_index = TDSS_fiber_indicies[0] #index of current TDSS fiber on plate (0 - 999)

    build_title = 'plate = '+str(currentplate.plate)+', mjd = '+str(currentplate.mjd)+', sourcetype = '+currentplate.source_type[TDSS_fiber_index]+', objtype = '+currentplate.obj_type[TDSS_fiber_index]+' \n Targeted by: '
    for target_class in get_ebosstarget_class(currentplate.eboss_target0[TDSS_fiber_index], currentplate.eboss_target1[TDSS_fiber_index], currentplate.eboss_target2[TDSS_fiber_index]):
        build_title += target_class+', '
    title = suptitle(build_title, fontsize=14)
    inspection_flags = [0]*len(TDSS_fiber_indicies)
    comments = ['nc']*len(TDSS_fiber_indicies)

    #make plots
    ##global ax1
    ax1 = subplot2grid((2,3), (0,0),colspan=1)
    SDSS_cutout=imread(urllib.urlretrieve('http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra='+str(currentplate.plug_ra[TDSS_fiber_index])+'&dec='+str(currentplate.plug_dec[TDSS_fiber_index])+'&scale=0.15&width=400&height=400&opt=GL')[0])
    ax1.implot = imshow(SDSS_cutout)
    ax1.xaxis.set_visible(False); ax1.yaxis.set_visible(False)
    ax1.set_title('ra = '+str(currentplate.plug_ra[TDSS_fiber_index])+', dec = '+str(currentplate.plug_dec[TDSS_fiber_index]), fontsize=14)

    ##global ax2
    # Updated 3/22/19 to lighten the color of the error bar around each flux measurement 
    ax2 = subplot2grid((2,3), (0,1),colspan=2)
    ticklabels = ax2.get_xticklabels()
    ticklabels.extend( ax2.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(12)
    ax2.set_ylabel(r'F$_\lambda$ [10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', size=12)
    ax2.set_xlabel(r'wavelength [$\AA$]', size=12)
    ax2.set_title('fiber = '+str(TDSS_fiber_index+1)+', class = '+currentplate.class0[TDSS_fiber_index]+', subclass = '+currentplate.subclass0[TDSS_fiber_index]+', z = '+str(round(currentplate.redshift[TDSS_fiber_index], 3)), fontsize=14)
    ax2.errorbar(currentplate.wavelength, smooth(currentplate.spectra[TDSS_fiber_index]), yerr=smooth(currentplate.fluxerr[TDSS_fiber_index]), ecolor='paleturquoise', color='k', capsize=0.)
    ax2.plot(currentplate.wavelength, currentplate.model[TDSS_fiber_index], 'r-')

    ##global PS1_matched_index; 
    PS1_matched_index = PS1_matched_indicies[0]
    ##global PS1_matched_dist; 
    PS1_matched_dist = PS1_matched_dists[0]
    ##global ax3
    ax3 = subplot2grid((2,3), (1,0),colspan=1)
    ax4 = subplot2grid((2,3), (1,1),colspan=2)
    if(PS1_matched_dist < 10):
        ax3.set_title('matched distance = '+str(round(PS1_matched_dist,2))+'"')
        ax3.contour(gr_ri_density, extent=contour_extent)
        ticklabels = ax3.get_xticklabels()
        ticklabels.extend( ax3.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(12)
        ax3.set_ylabel(r'$g - r$', size=14)
        ax3.set_xlabel(r'$u - g$', size=14)
        ax3.plot(PS1_lc.PS1_sdss_ps1[PS1_matched_index][0] - PS1_lc.PS1_sdss_ps1[PS1_matched_index][1], PS1_lc.PS1_sdss_ps1[PS1_matched_index][1] - PS1_lc.PS1_sdss_ps1[PS1_matched_index][2], 'r*', markersize=12)

        ##global ax4
        # Modified 3/22/19 to add warning title if PS1 match is more than 10 arcseconds away
        g_mag, r_mag, i_mag, g_mjd, r_mjd, i_mjd, g_err, r_err, i_err = PS1_lc.get_3pi_lightcurve() # get lightcurves of current PS1_matched_index
        ticklabels = ax4.get_xticklabels()
        ticklabels.extend( ax4.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(12)
        ax4.set_ylabel(r'PS1 mag', size=12)
        ax4.set_xlabel(r'MJD', size=12)
        ax4 = gca()
        ax4.invert_yaxis()
        maxx = 0; minx = 999999.
        if len(g_mag)>0: 
            ax4.errorbar(g_mjd, g_mag, yerr = g_err, ecolor='g', color='g', fmt='o', label=r'$g$')
            if np.min(g_mjd)<minx: minx = np.min(g_mjd)
            if np.max(g_mjd)>maxx: maxx = np.max(g_mjd)
        if len(r_mag)>0: 
            ax4.errorbar(r_mjd, r_mag, yerr = r_err, ecolor='r', color='r', fmt='o', label=r'$r$')
            if np.min(r_mjd)<minx: minx = np.min(r_mjd)
            if np.max(r_mjd)>maxx: maxx = np.max(r_mjd)
        if len(i_mag)>0: 
            ax4.errorbar(i_mjd, i_mag, yerr = i_err, ecolor='b', color='b', fmt='o', label=r'$i$')
            if np.min(i_mjd)<minx: minx = np.min(i_mjd)
            if np.max(i_mjd)>maxx: maxx = np.max(i_mjd)
        ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][0],PS1_lc.PS1_sdss_ps1[PS1_matched_index][0]], 'g--')
        ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][1],PS1_lc.PS1_sdss_ps1[PS1_matched_index][1]], 'r--')
        ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][2],PS1_lc.PS1_sdss_ps1[PS1_matched_index][2]], 'b--')
        ax4.legend(loc=3, prop={'size':12}, ncol=3, numpoints=1, columnspacing = 0.2, handletextpad=.01, borderpad=.02)

    callback = callback_handler()
    axaccept = axes([0.7, 0.02, 0.1, 0.055])
    axflag = axes([0.81, 0.02, 0.1, 0.055])
    axback = axes([0.08, 0.02, 0.1, 0.055])
    axrestframe = axes([0.51, 0.02, 0.1, 0.055])
    axobsframe = axes([0.40, 0.02, 0.1, 0.055])
    axcomment = axes([0.19, 0.02, 0.1, 0.055])

    baccept = Button(axaccept, 'Accept')
    bflag = Button(axflag, 'Flag')
    bback = Button(axback, 'Back')
    brestframe = Button(axrestframe, 'Rest Frame')
    bobsframe = Button(axobsframe, 'Obs Frame')
    bcomment = Button(axcomment, 'Comment')

    baccept.on_clicked(callback.accept)
    bflag.on_clicked(callback.flag)
    bback.on_clicked(callback.back)
    brestframe.on_clicked(callback.restframe)
    bobsframe.on_clicked(callback.obsframe)
    bcomment.on_clicked(callback.comment)
    show()

elif len(sys.argv) > 3:
    #global TDSS_fiber_index  #index of current TDSS fiber on plate (0 - 999)

    plate = str(sys.argv[1]); mjd = str(sys.argv[2]); TDSS_fiber_indicies = asarray([int(x)-1 for x in sys.argv[3:]]); TDSS_fiber_index =  TDSS_fiber_indicies[0]
    platefile = urllib.urlretrieve('http://data.sdss.org/sas/ebosswork/eboss/spectro/redux/v5_10_10/'+plate+'/spPlate-'+plate+'-'+mjd+'.fits')
    platefile = pyfits.open(platefile[0])
    spZbestfile = urllib.urlretrieve('http://data.sdss.org/sas/ebosswork/eboss/spectro/redux/v5_10_10/'+plate+'/v5_10_10/spZbest-'+plate+'-'+mjd+'.fits')
    spZbestfile = pyfits.open(spZbestfile[0])
    currentplate = platespectra(platefile, spZbestfile)
    PS1_lc = PS1_3pi_lightcurves(max(currentplate.plug_ra), min(currentplate.plug_ra), max(currentplate.plug_dec), min(currentplate.plug_dec))
    gr_ri_density, contour_extent = PS1_lc.ug_gr_contours()

    #find plate indices (0-999) of the TDSS targets by matching to PS1 targets file
    #global PS1_matched_indicies, PS1_matched_dists, 
    PS1_matched_indicies = []; PS1_matched_dists = []
    for current_fiber in TDSS_fiber_indicies:
        if ((currentplate.eboss_target1[current_fiber] & 2**30) != 0): #if TDSS target
            current_PS1_matched_index, current_PS1_matched_dist = PS1_lc.find_PS1_counterpart(currentplate.plug_ra[current_fiber], currentplate.plug_dec[current_fiber])
            PS1_matched_dists.append(current_PS1_matched_dist)
            PS1_matched_indicies.append(current_PS1_matched_index)
    PS1_matched_dists = asarray(PS1_matched_dists); PS1_matched_indicies = asarray(PS1_matched_indicies)

    # set plotting canvas
    figure(figsize=(13, 9), dpi=80)
    subplots_adjust(bottom=.13, wspace = .4, hspace=.25, left=.07, right=.95, top=.88)
    #global title, inspection_flags #flags from visual inspection, 1=accept, 2=flag

    build_title = 'plate = '+str(currentplate.plate)+', mjd = '+str(currentplate.mjd)+', sourcetype = '+currentplate.source_type[TDSS_fiber_index]+', objtype = '+currentplate.obj_type[TDSS_fiber_index]+' \n Targeted by: '
    for target_class in get_ebosstarget_class(currentplate.eboss_target0[TDSS_fiber_index], currentplate.eboss_target1[TDSS_fiber_index], currentplate.eboss_target2[TDSS_fiber_index]):
        build_title += target_class+', '
    title = suptitle(build_title, fontsize=14)
    inspection_flags = [0]*len(TDSS_fiber_indicies)

    #make plots
    #global ax1
    ax1 = subplot2grid((2,3), (0,0),colspan=1)
    SDSS_cutout=imread(urllib.urlretrieve('http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra='+str(currentplate.plug_ra[TDSS_fiber_index])+'&dec='+str(currentplate.plug_dec[TDSS_fiber_index])+'&scale=0.15&width=400&height=400&opt=GL')[0])
    ax1.implot = imshow(SDSS_cutout)
    ax1.xaxis.set_visible(False); ax1.yaxis.set_visible(False)
    ax1.set_title('ra = '+str(currentplate.plug_ra[TDSS_fiber_index])+', dec = '+str(currentplate.plug_dec[TDSS_fiber_index]), fontsize=14)

    #global ax2

    ax2 = subplot2grid((2,3), (0,1),colspan=2)
    ticklabels = ax2.get_xticklabels()
    ticklabels.extend( ax2.get_yticklabels() )
    for label in ticklabels:
        label.set_fontsize(12)
    ax2.set_ylabel(r'F$_\lambda$ [10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', size=12)
    ax2.set_xlabel(r'wavelength [$\AA$]', size=12)
    ax2.set_title('fiber = '+str(TDSS_fiber_index+1)+', class = '+currentplate.class0[TDSS_fiber_index]+', subclass = '+currentplate.subclass0[TDSS_fiber_index]+', z = '+str(round(currentplate.redshift[TDSS_fiber_index], 3)), fontsize=14)
    ax2.errorbar(currentplate.wavelength, smooth(currentplate.spectra[TDSS_fiber_index]), yerr=smooth(currentplate.fluxerr[TDSS_fiber_index]), ecolor='paleturquoise', color='k', capsize=0.)
    ax2.plot(currentplate.wavelength, currentplate.model[TDSS_fiber_index], 'r-')

    #global PS1_matched_index; 
    PS1_matched_index = PS1_matched_indicies[0]
    #global PS1_matched_dist; 
    PS1_matched_dist = PS1_matched_dists[0]
    #global ax3
    ax3 = subplot2grid((2,3), (1,0),colspan=1)
    ax4 = subplot2grid((2,3), (1,1),colspan=2)    
    if(PS1_matched_dist < 10):
        ax3.set_title('matched distance = '+str(round(PS1_matched_dist,2))+'"')
        ax3.contour(gr_ri_density, extent=contour_extent)
        ticklabels = ax3.get_xticklabels()
        ticklabels.extend( ax3.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(12)
        ax3.set_ylabel(r'$g - r$', size=14)
        ax3.set_xlabel(r'$u - g$', size=14)
        ax3.plot(PS1_lc.PS1_sdss_ps1[PS1_matched_index][0] - PS1_lc.PS1_sdss_ps1[PS1_matched_index][1], PS1_lc.PS1_sdss_ps1[PS1_matched_index][1] - PS1_lc.PS1_sdss_ps1[PS1_matched_index][2], 'r*', markersize=12)

        #global ax4
        g_mag, r_mag, i_mag, g_mjd, r_mjd, i_mjd, g_err, r_err, i_err = PS1_lc.get_3pi_lightcurve() # get lightcurves of current PS1_matched_index
        ticklabels = ax4.get_xticklabels()
        ticklabels.extend( ax4.get_yticklabels() )
        for label in ticklabels:
            label.set_fontsize(12)
        ax4.set_ylabel(r'PS1 mag', size=12)
        ax4.set_xlabel(r'MJD', size=12)
        ax4 = gca()
        ax4.invert_yaxis()
        maxx = 0; minx = 999999.
        if len(g_mag)>0: 
            ax4.errorbar(g_mjd, g_mag, yerr = g_err, ecolor='g', color='g', fmt='o', label=r'$g$')
            if np.min(g_mjd)<minx: minx = np.min(g_mjd)
            if np.max(g_mjd)>maxx: maxx = np.max(g_mjd)
        if len(r_mag)>0: 
            ax4.errorbar(r_mjd, r_mag, yerr = r_err, ecolor='r', color='r', fmt='o', label=r'$r$')
            if np.min(r_mjd)<minx: minx = np.min(r_mjd)
            if np.max(r_mjd)>maxx: maxx = np.max(r_mjd)
        if len(i_mag)>0: 
            ax4.errorbar(i_mjd, i_mag, yerr = i_err, ecolor='b', color='b', fmt='o', label=r'$i$')
            if np.min(i_mjd)<minx: minx = np.min(i_mjd)
            if np.max(i_mjd)>maxx: maxx = np.max(i_mjd)
        ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][0],PS1_lc.PS1_sdss_ps1[PS1_matched_index][0]], 'g--')
        ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][1],PS1_lc.PS1_sdss_ps1[PS1_matched_index][1]], 'r--')
        ax4.plot([minx, maxx], [PS1_lc.PS1_sdss_ps1[PS1_matched_index][2],PS1_lc.PS1_sdss_ps1[PS1_matched_index][2]], 'b--')
        ax4.legend(loc=3, prop={'size':12}, ncol=3, numpoints=1, columnspacing = 0.2, handletextpad=.01, borderpad=.02)

    callback = callback_handler()
    axgoforward = axes([0.81, 0.02, 0.1, 0.055])
    axgoback = axes([0.7, 0.02, 0.1, 0.055])
    axrestframe = axes([0.51, 0.02, 0.1, 0.055])
    axobsframe = axes([0.40, 0.02, 0.1, 0.055])
    bgoforward = Button(axgoforward, 'Next')
    bgoback = Button(axgoback, 'Back')
    brestframe = Button(axrestframe, 'Rest Frame')
    bobsframe = Button(axobsframe, 'Obs Frame')
    bgoforward.on_clicked(callback.goforward)
    bgoback.on_clicked(callback.goback)
    brestframe.on_clicked(callback.restframe)
    bobsframe.on_clicked(callback.obsframe)
    show()
