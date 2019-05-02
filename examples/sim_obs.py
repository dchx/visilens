import numpy as np
import os,sys; sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
import visilens as vl
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.cosmology import Planck15

def sim_obs(outfits, Lens, Source, npixside=200, center=(0,0), res=0.01, gridunit='arcsec', cosmo=Planck15):
	'''
	Simulate observed lensed sources.
	-----
	Parameters
	    outfits: output fits file name
	    Lens: SIELens or ExternalShear or list of Lenses
	    Source: GaussSource, SersicSource, PointSource or list of Sources
	        source position: relative to the first Lens.
	    npixside: (int) number of pixesl of generated image; default 200
	    center: size 2 (list-like) center of generated image in (arcsec) or
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

	# compute grids
	sidelen = res * (npixside - 1) # image side length from starting pixel to ending pixel in gridunit
	gridlim = [center[0]-sidelen/2., center[0]+sidelen/2., center[1]-sidelen/2., center[1]+sidelen/2.]
	x = np.linspace(gridlim[0], gridlim[1], npixside)
	y = np.linspace(gridlim[2], gridlim[3], npixside)
	x, y = np.meshgrid(x, y)
	icen = (npixside + 1)/2. # center x/y pixel index (start from 1)

	# perform lensing
	xsource, ysource = vl.LensRayTrace(x,y,lenses,Dd,Ds,Dds)
	imlensed = np.zeros(xsource.shape)
	for source in sources:
		imlensed += vl.SourceProfile(xsource,ysource,source,lenses)
	
	# write to fits file
	hdu = fits.PrimaryHDU(imlensed)
	for i,AXIS in enumerate(['1', '2']):
		hdu.header['CUNIT'+AXIS] = 'deg'
		hdu.header['CRPIX'+AXIS] = icen # reference pixel
		hdu.header['CRVAL'+AXIS] = (center[i] * u.Unit(gridunit)).to(u.degree).value # reference pixel coordinate
		hdu.header['CDELT'+AXIS] = (res * u.Unit(gridunit)).to(u.degree).value # pixel scale
	#plt.imshow(imlensed)
	#plt.show()
	hdu.writeto(outfits,overwrite=True)
	print "Saved: ", outfits

if __name__ == '__main__':
	lens = vl.SIELens(\
	    z = 0.8,\
	    x = 0.,\
	    y = 0.,\
	    M = 1e11,\
	    e = 0.,\
	    PA = 0.) # M in Msun, PA in degrees east of north.
	'''
	source = vl.SersicSource(\
	    z = ,\
	    xoff = ,\
	    yoff = ,\
	    flux = ,\
	    majax = ,\
	    index = ,\
	    axisratio = ,\
	    PA = ) # x/yoff relative to the first lens, flux is total integrated flux in Jy
	'''
	source = vl.GaussSource(\
	    z = 5.6559,\
	    xoff = 0.,\
	    yoff = 0.,\
	    flux = 0.01,\
	    width = 0.1) # x/yoff relative to the first lens, flux is total integrated flux in Jy
	'''
	source = vl.PointSource(\
	    z = 5.6559,\
	    xoff = 0.,\
	    yoff = 0.,\
	    flux = 0.01) # x/yoff relative to the first lens, flux is total integrated flux in Jy
	'''
	path = '/home/dcx/dcx/jwst/ripplestest/models/'
	path = '/astro/homes/dcx/dcxroot/sptalma/visilens/visilens/'
	out = path + 'simed_image.fits'
	sim_obs(out, lens, source)