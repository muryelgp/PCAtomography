
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from astropy import units, table, wcs
from astropy.io import fits
from scipy.interpolate import interp1d , UnivariateSpline
import scipy as sp
from scipy import linalg

class IFUcube:
	"""
    A class for applying Principal Component Analyses (PCA) Tomography in Integral Field Unit (IFU) spectroscopy data cubes. See a description of the method in Steiner et al. 2009 (https://arxiv.org/pdf/0901.2266).
    """
	def __init__(self,fname, scidata='SCI', primary='PRIMARY', spectral_dimension =3, emission_line_mask = None ):
		"""
		Instantiates the class IFUcube.
	   
		Parameters
		----------
		fname : string
            Name of the FITS file containing the IFU datacube. This
            should be the standard output from the FITS file.

		scidata: integer or string
            Extension of the FITS file containing the scientific data.

        primary: integer or string
            Extension of the FITS file containing the basic header.	

		Returns
		-------
		Nothing.
		"""		
		self.fitsfile = fname
		self.tomograms = None
		self.eigenvectors = None
		self.eigenvalues = None


		with fits.open(fname) as hdu:
			self.data = hdu[scidata].data
			self.header = hdu[primary].header
			self.header_data = hdu[scidata].header
			self.wcs = wcs.WCS(self.header_data)
			

			labels = ['TOMOGRAMS', 'EIGENVECTORS', 'EIGENVALUES']

			try:	
				self.tomograms = hdu[labels[0]].data
				self.eigenvectors = hdu[labels[1]].data
				self.eigenvalues = hdu[labels[2]].data
			except:
				pass

		self.wl = self.wcs.sub((spectral_dimension,)).wcs_pix2world(np.arange(len(self.data)), 0)[0]

		if self.wcs.wcs.cunit[2] == units.m:
			self.wl *= 1e+10


		if emission_line_mask  is not None:
			self.em_mask = 1*(emission_line_mask)
		else:
			self.em_mask = None	


				
	def run_pca(self, components = 'all', emission_line_mask= None, emission_line_treatment = 'interpolate' ,write_fits = False, out_ext = '_PCA.fits', normalization = False, norm_win = [None,None]):
		"""
		Runs the PCA Tomography algoritm in the IFUcube (see detail in Steiner et al. 2009, https://arxiv.org/pdf/0901.2266.


		Parameters
		--------
		components: integer or string
            How many components to be saved. The standard is 'all', but it can become too large for large datacubes (e.g. MUSE datacubes).
            The other option is to select the number (integer) of components to be saved, e.g. components = 100.

      	emission_line_mask: list of array
      		The emission lines to be masked. It has to be a Boolean array (False value for the emission lines) with the same size as the spectral dimension of the datacube.

      	emission_line_treatment: string
      		If a 'emission_line_mask' is passed, 'emission_line_treatment' designs the treatment of the emission lines, wheter they will interpolated by 'best-fitting' curve (emission_line_treatment = 'interpolate')
      		or they will be assign as nan's (emission_line_treatment = 'nan'). the standart is 'interpolate'.

      	write_fits: Boolean
      		Whether to save the results in a fits file. It can be read after with 'red_pca()' method.

      	out_ext: string
      		The extension to the fits file to be saved. Only applies if 'write_fits = True'.

      	normalization: Boolean
      		Whether to normalize the spectra before applying the PCA Tomography (NOT WORKING, TO BE IMPLEMENTED!).

      	Returns
		-------
		Nothing.
		"""

		if emission_line_mask  is not None:
			self.em_mask = 1*(emission_line_mask)
		else:
			self.em_mask = None




		print('--------------------------------------------')
		print('IT CAN TAKE SOME TIME AND SOME MEMORY!')
		print('--------------------------------------------')

		self.new_data = deepcopy(self.data)

		##Masking Emission lines
		if self.em_mask  is not None:
			if np.shape(np.array(self.em_mask))[0] == self.new_data.shape[0]:
				

				self.em_mask[0] = True
				self.em_mask[-1] = True

				if(emission_line_treatment == 'interpolate'):
					begin = []
					end = []

					for i in range(len(self.em_mask) -1):
						if (self.em_mask[i+1] -  self.em_mask[i]) == -1:
							begin.append(i)
						if	(self.em_mask[i+1] -  self.em_mask[i]) == 1:
							end.append(i)

					for j in range(len(begin)):
						
						for m in range(self.new_data.shape[1]):
							for n in range(self.new_data.shape[2]):
								xs = [self.wl[begin[j]] , self.wl[end[j]]]
								ys = [self.data[begin[j],m,n]  , self.data[end[j],m,n] ]
								interp = interp1d(xs,ys)
								self.new_data[begin[j]:(end[j]+  1),m,n] = interp(self.wl[begin[j]:(end[j]+1)])


				elif(emission_line_treatment == 'nan'):
					self.new_data[:,:,self.em_mask] = 0
				else:
					raise RuntimeError("emission_line_treatment needs tobe either 'interpolate' or 'nan'")	
			else:
				raise RuntimeError("'emission_line_mask' {} needs to have the same shape as the spectral dimension of the IFUcube {}".format(self.new_data.shape[0],np.shape(np.array(self.em_mask))))





		#applying nan mask
		self.nan_mask = ~np.isfinite(self.data)
		self.new_data[self.nan_mask] = 0.0

		self.zones = self._gen_zones()
		n = len(self.zones)

		#Median spectra
		Q_lambda = ((self.new_data.sum(axis=1)).sum(axis=1))/n

		#subtractin mean spectra
		I_ij_lambda = (self.new_data.T) - (Q_lambda)
		I_ij_lambda = I_ij_lambda.T
		
		Q_lambda = None

		#from 3D to 2D
		I_B_lambda = np.zeros((n ,I_ij_lambda.shape[0]))
		n=0
		for i in self.zones:
			I_B_lambda[n , :] = I_ij_lambda[:,i[1], i[0]] 
			n=n+1
	
		I_ij_lambda = None
		#covariance Matrix
		C_cov = (sp.dot(I_B_lambda.transpose(), I_B_lambda)) / (n-1)

		#solving eigen-problem
		w_k,E_k = linalg.eigh(C_cov)








		#Organazing by eigenvalue
		S = sp.argsort(w_k)[::-1]
		self.eigenvalues = w_k[S]
		self.eigenvectors = E_k[:, S]

		#New coordenats system
		T_B_k = sp.dot(I_B_lambda, self.eigenvectors)

		

		if (components != 'all'):
			try:
				n_comp = int(components)
			except ValueError:
				print("'components' needs to be 'all' or an integer")
			
			self.tomograms= np.zeros((n_comp,  self.new_data.shape[1], self.new_data.shape[2]))

			for n, i in enumerate(self.zones):
				self.tomograms[:,i[1], i[0]]  = T_B_k[n, :(n_comp)]
			T_B_k[:] = None

			self.eigenvectors = self.eigenvectors[:, :n_comp]
			self.eigenvalues = self.eigenvalues[:n_comp]

		elif(components == 'all'):

			self.tomograms= np.zeros(np.shape(self.new_data))
			for n, i in enumerate(self.zones):
				self.tomograms[:,i[1], i[0]]  = T_B_k[n, :]
			T_B_k[:] = None





		if write_fits:
			hdu = fits.PrimaryHDU([], header = self.header)
			data  = fits.ImageHDU(self.data, name = 'SCI', header = self.header_data)
			tom  = fits.ImageHDU(self.tomograms, name = 'Tomograms')
			eigenvectors = fits.ImageHDU(self.eigenvectors, name = 'Eigenvectors')
			eigenvalues = fits.ImageHDU(self.eigenvalues, name = 'Eigenvalues')
			hdul = fits.HDUList([hdu, data,tom,eigenvectors, eigenvalues])
			hdul.writeto(self.fitsfile[:-5] + out_ext)

		
		print('DONE!')

	def _gen_zones(self):
		return np.array([[i,j] for i in range(self.data.shape[2]) for j in range(self.data.shape[1])])

	@staticmethod
	def read_pca(fname, scidata='SCI', primary='PRIMARY', emission_line_mask= None, spectral_dimension=3):
		"""
		Method for reading fits file created with 'run_pca()' method.

		"""
		temp_obj = IFUcube(fname, emission_line_mask = emission_line_mask)
		return temp_obj

	def plot_tomogram(self,component, contrast=1):
		"""
		Plots the tomogram and its respective Eigenvector for a given component.


		Parameters
		--------
		component: integer
            Which component to be plotted. The component numbers starts in 1. 

      	contrast: integer
      		The colour contrast for the matplotlib pcolormesh plot.

		"""


		data = self.tomograms[component -1,:, ]
		eigh = self.eigenvectors[:, component -1]
		if self.em_mask is not None:
			eigh[~self.em_mask] = np.nan

		fig = plt.figure(figsize = (12,4))
		ax1 = plt.subplot2grid((3, 12), (0, 0), rowspan=3, colspan=3)
		y,x = np.indices(np.shape(data))
		y,x = y-np.max(y)/2 ,x- np.max(x)/2
		
		if ( eigh[(abs(eigh) == np.nanmax(abs(eigh)))]  < 0):
			eigh = eigh *-1.0
			data = data*-1.0
		img = ax1.pcolormesh(x,y,data, vmin = np.percentile(data, contrast), vmax = np.percentile(data, 100-contrast))
		ax1.set_title('Tomogram ' + str(component))

		ax2 = plt.subplot2grid((3, 12), (0, 4), rowspan=3, colspan=8)
		ax2.plot(self.wl, eigh, label = "eigen-spectra " + str(component))
		ax2.set_ylabel(r"E$_{} $".format(str(component)) + ' [Arbitrary Units]')
		ax2.set_xlabel(r'Wavelenght [$\AA$]')
		plt.legend()
		plt.show()
		
	def clean_data(self,components, fit_spline = True, smoothing_factor=0.1, write_fits = False):
		"""
		Removes choosen components of the datacube, it is particularlly useful in 'Instrumental Fingerprint' removal from IFU spectroscopy data cubes (see the details in Menezes et al. 2014, https://arxiv.org/abs/1401.7078).


		Parameters
		--------
		components: integer or list
            Which components to be remove. It can be one component (e.g. components = 5) or a list (e.g. components = [22,30,45])

      	fit_spline: Boolean
      		Wheter to fit or not a spline to the Eigenvector before removing it. The standart it True.

      	smoothing_factor: float
      		The smoothing factor of the spline. Only applies if 'fit_spline = True'.

      	write_fits: Boolean
      		Whether to save the results in a fits file. It can be read after with 'red_pca()' method.

      	out_ext: string
      		The extension to the fits file to be saved. Only applies if 'write_fits = True'. See the documentation in scipy manual: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html.

      	write_fits: Boolean
      		Whether to add a 'CLEAN DATA' extension in the IFU data cube.

      	Returns
		-------
		clean_data: array
			The array containing the data with the component removed.
		"""



		clean_data = self.data


		if isinstance(components, int):
			data = self.tomograms[components -1,:, ]
			eigh = self.eigenvectors[:, components -1]
			if ( eigh[(abs(eigh) == np.nanmax(abs(eigh)))]  < 0):
				eigh = eigh *-1.0
				data = data*-1.0

			if self.em_mask is not None:
				eigh[~self.em_mask] = np.nan

			if fit_spline:
				spline = UnivariateSpline(self.wl, eigh,s=smoothing_factor)
				plt.plot(self.wl, eigh, label = "eigen-spectra " + str(components))
				plt.plot(self.wl,spline(self.wl), label = "spline fit"  )
				plt.show()

				eigen_cube = data * spline(self.wl)[:, np.newaxis, np.newaxis]
				clean_data = clean_data - eigen_cube
				
			elif(~fit_spline):
				eigen_cube = data * eigh[:, np.newaxis, np.newaxis]
				clean_data = clean_data - eigen_cube
				

		if isinstance(components, list):
			for i in components:
				data = self.tomograms[i -1,:, ]
				eigh = self.eigenvectors[:, i -1]
				if ( eigh[(abs(eigh) == np.nanmax(abs(eigh)))]  < 0):
					eigh = eigh *-1.0
					data = data*-1.0

				if fit_spline:
					spline = UnivariateSpline(self.wl, eigh,s=smoothing_factor)
					plt.plot(self.wl, eigh, label = "eigen-spectra " + str(i))
					plt.plot(self.wl,spline(self.wl), label = "spline fit"  )
					plt.show()
					eigen_cube = data * spline(self.wl)[:, np.newaxis, np.newaxis]
					clean_data = clean_data - eigen_cube
				elif(~fit_spline):
					eigen_cube = data * eigh[:, np.newaxis, np.newaxis]
					clean_data = clean_data - eigen_cube
					
			

		if write_fits:
			hdul = fits.open(self.fitsfile)
			hdul.append(fits.ImageHDU(clean_data, name = 'CLEAN SCI', header = self.header_data))
			hdul.writeto(self.fitsfile, overwrite = True)


		return clean_data



	def plot_scree_test(self,components = 15, log_scale = True, scree_line = 0.1):
		"""
		Plots the scree test for the PCA Tomography (see details in Steiner et al. 2009, https://academic.oup.com/mnras/article/395/1/64/1078341).


		Parameters
		--------
		components: integer
            Which components to be ploted.

      	log_scale: Boolean
      		Whether to show the Covariance axis (y) in log scale.

      	scree_line: float
      		Where to draw the reference line for the scree test.

		"""

		if components > len(self.eigenvalues):
			components = int(len(self.eigenvalues))


		var = (self.eigenvalues/np.sum(self.eigenvalues))[:(components)] * 100

		component =  np.arange(1, components+1, 1)
		plt.xlim(0,components+1)

		plt.scatter(component, var)
		plt.ylabel(r'Variance (%)')
		plt.xlabel(r'Eigenvector number')
		plt.xticks(component)



		scale = var[0] - var[-1]
		if log_scale:
			plt.yscale('log')
			
			plt.hlines(scree_line, xmin = 0, xmax=components+1,color= 'red',  linestyle = '--', linewidth = 0.5, label = 'Variance = ' +str(scree_line) +' %')
		else:
			plt.hlines(scree_line, xmin = 0, xmax=components+1,color= 'red', linestyle = '--', linewidth = 0.5, label = 'Variance = ' +str(scree_line) +' %')
			plt.hlines(0, xmin = 0, xmax=components+1,color= 'black', linewidth = 0.5, label = 'Variance = 0.0 %')

		plt.legend()
		plt.show()		
			


	def get_tomogram(self, component):

		tomogram = self.tomograms[component -1,:, ]
		return tomogram

	def get_eigenvector(self, component):

		eigh = self.eigenvectors[:, component -1]
		return eigh



if __name__ == "__main__":
	
	a = 1

	'''
	cube =IFUcube('/mnt/d/cubos_gmos/ngc3081_HYPERCUBE.fits')
	em = ((cube.wl>6580) & (cube.wl<6650)) or (((cube.wl>6580) & (cube.wl<6650)))


	cube.run_pca(emission_line_mask= ~em)
	cube.plot_tomogram(1)

	#cube = IFUcube.read_pca('/mnt/d/cubos_gmos/ngc3081_HYPERCUBE_PCA.fits')
	#cube.plot_scree_test()
	'''
