from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='PCAtomography',
    version='1.0.0',
    description='Package for applying Principal Component Analyses (PCA) Tomography in Integral Field Unit (IFU) spectroscopy data cubes',
    long_description=readme,
    author='Muryel Guolo Pereira',
    author_email='muryel@astro.ufsc.br',
    url='https://github.com/muryelgp/PCAtomography',
    download_url = 'https://github.com/muryelgp/PCAtomography/archive/master.zip',
    license=license,
    keywords = ['methods', 'Astronomy', 'PCA'],
    packages=['PCAtomography'],
    install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'astropy',
      ],
)
