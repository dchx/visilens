import numpy as np
import os,sys 
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import Planck15
import scipy.signal as sig
from astropy.modeling.models import Moffat2D,Gaussian2D
from lensing import LensRayTrace
from class_utils import *
from calc_likelihood import SourceProfile

__all__ = ['sim_obs']

def write_fits(im, tosave, refpix, refcoo, res, gridunit='arcsec'):
	'''
	# write to fits file
	refpix: reference pixel
	refcoo: reference coordinate (gridunit)
	res: resolution (gridunit/pixel)
	gridunit: defalut 'arcsec'
	'''
	hdu = fits.PrimaryHDU(im)
	for iax,AXIS in enumerate(['1', '2']):
		hdu.header['CUNIT'+AXIS] = 'deg'
		hdu.header['CRPIX'+AXIS] = refpix # reference pixel
		hdu.header['CRVAL'+AXIS] = (refcoo[iax] * u.Unit(gridunit)).to(u.degree).value # reference pixel coordinate
		hdu.header['CDELT'+AXIS] = (res * u.Unit(gridunit)).to(u.degree).value # pixel scale
	hdu.writeto(tosave, overwrite=True)
	print "Saved: ", tosave

def add_noise(image, sigma=None, sky_level=None, sky_rela=0.2):
	'''
	Add poisson noise on image

	Paramaters
	-------------
	image: 2-d array image
	sigma: if None, add poisson noise, else add gaussian noise
	sky_level: absolute sky level in image units
	sky_rela: sky level relative to maximum image flux

	Return
	-------------
	image with noise added
	rms map
	'''
	if sky_level != None: sky = sky_level
	else: sky = np.max(image) * sky_rela
	im_wsky = image + sky
	if sigma==None: # add poisson noise
		noise = np.random.poisson(lam=abs(im_wsky)) - im_wsky
		im_noise = noise.std() * np.ones(image.shape)
	else: # add gaussian noise
		noise = np.random.normal(scale=sigma, size=image.shape)
		im_noise = sigma * np.ones(image.shape)
	return image + noise, im_noise

def convolve_psf(image, res, savepsf=False, model='moffat', psf_width=0.1, noise_sigma=None, gridunit='arcsec'):
	'''
	image: 2d array
	res: (arcsec/pix) image resolution
	model: moffat or gaussian
	psf_width: (arcsec) default 0.1
	'''
	psf_len = 5 * psf_width # (arcsec) psf image length
	grid_right = np.arange(0.,psf_len/2.+res,res)
	grid = np.hstack([(grid_right-psf_len/2.)[:-1],grid_right])
	x, y = np.meshgrid(grid, grid)
	if model == 'moffat': psf = Moffat2D(gamma = psf_width)(x,y)
	elif 'gauss' in model: psf = Gaussian2D(x_stddev = psf_width, y_stddev = psf_width)(x,y)
	psf = psf / np.sum(psf) # normalize
	image_out = sig.fftconvolve(image,psf,mode='same')

	# write psf to fits
	psf, _ = add_noise(psf, noise_sigma) # add noise for output psf
	if savepsf:
		psf_icen = (len(grid) + 1)/2. # center x/y pixel index (start from 1)
		write_fits(psf, savepsf, refpix=psf_icen, refcoo=(0,0), res=res, gridunit=gridunit)
	return image_out

def sim_obs(outprefix, Lens, Source, npixside=200, center=(0,0), res=0.01, gridunit='arcsec', src_center=(0,0), src_npixside=200, src_res=0.01, psf_model='moffat', psf_width=0.1, noise_sigma=None, cosmo=Planck15):
	'''
	Simulate observed lensed sources.
	-----
	Parameters
	    outprefix: output fits file prefix; output=outprefix+'_image/source.fits'
	    Lens: SIELens or ExternalShear or list of Lenses
	    Source: GaussSource, SersicSource, PointSource or list of Sources
	        source position: relative to the first Lens.
	    npixside: (int) number of side pixels of generated image; default 200
	    center: size 2 (list-like) define center of generated image in (arcsec) or
	            the same units as in Lens and Source (gridunit); default (0,0)
	    res: [gridunit/pix] pixel resolution; default 0.01
	    gridunit: default 'arcsec'
	Assuming all lenses and sources have the same redshift.
	'''
	# force lens and source to be arrays
	lenses = list(np.array([Lens]).flatten())
	sources = list(np.array([Source]).flatten())

	# distances
	try: zLens = lenses[0].z
	except: raise Exception("The first lens must be an SIELens")
	zSource = sources[0].z
	Dd = cosmo.angular_diameter_distance(zLens).value
	Ds = cosmo.angular_diameter_distance(zSource).value
	Dds = cosmo.angular_diameter_distance_z1z2(zLens,zSource).value

	# wrap image and source parameters
	npixsides = [npixside, src_npixside]
	reses = [res, src_res]
	centers = [center, src_center]
	outsuffixes = ['_image', '_source']

	for i in range(len(npixsides)): # loop for source/image
		# compute grids
		sidelen = reses[i] * (npixsides[i] - 1) # image side length from starting pixel to ending pixel in gridunit
		gridlim = [centers[i][0]-sidelen/2., centers[i][0]+sidelen/2., centers[i][1]-sidelen/2., centers[i][1]+sidelen/2.]
		x = np.linspace(gridlim[0], gridlim[1], npixsides[i])
		y = np.linspace(gridlim[2], gridlim[3], npixsides[i])
		x, y = np.meshgrid(x, y)
		icen = (npixsides[i] + 1)/2. # center x/y pixel index (start from 1)

		# perform lensing
		if i==0: x, y = LensRayTrace(x,y,lenses,Dd,Ds,Dds) # image plane coords -> source plane coords
		im = np.zeros(x.shape)
		for source in sources: im += SourceProfile(x,y,source,lenses)

		# convolve psf, add noise for image (source not)
		if i==0:
			im = convolve_psf(im, reses[i], model=psf_model, psf_width=psf_width, noise_sigma=noise_sigma, gridunit=gridunit, savepsf=outprefix+outsuffixes[i]+'_psf.fits')
			im, rms = add_noise(im, noise_sigma)

		# write to fitsfile
		tosave = outprefix+outsuffixes[i]+'.fits'
		write_fits(im, tosave, refpix=icen, refcoo=centers[i], res=reses[i], gridunit=gridunit)
		if i==0: 
			tosave_rms = outprefix+outsuffixes[i]+'_rms.fits'
			write_fits(rms, tosave_rms, refpix=icen, refcoo=centers[i], res=reses[i], gridunit=gridunit)

		# have a glance
		plt.imshow(im, extent=gridlim, origin='lower')
		plt.title(os.path.basename(tosave))
		plt.show()

if __name__ == '__main__':
	lens = SIELens(\
	    z = 0.5,\
	    x = 0.,\
	    y = 0.,\
	    M = 10**(10* 1.1),\
	    e = 0.5,\
	    PA = 0.) # M in Msun, PA in degrees east of north.
	'''
	source = SersicSource(\
	    z = ,\
	    xoff = ,\
	    yoff = ,\
	    flux = ,\
	    majax = ,\
	    index = ,\
	    axisratio = ,\
	    PA = ) # x/yoff relative to the first lens, flux is total integrated flux in Jy
	'''
	source = GaussSource(\
	    z = 1.,\
	    xoff = 0.1,\
	    yoff = 0.1,\
	    flux = 0.1,\
	    width = 0.1) # x/yoff relative to the first lens, flux is total integrated flux in Jy
	'''
	source = PointSource(\
	    z = 5.6559,\
	    xoff = 0.,\
	    yoff = 0.,\
	    flux = 0.01) # x/yoff relative to the first lens, flux is total integrated flux in Jy
	'''
	path = '/home/dcx/dcx/jwst/ripplestest/models/'
	out = path + 'simed_comp'
	sim_obs(out, lens, source)
