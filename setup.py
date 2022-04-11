from setuptools import setup

setup(name='specPeak',
      version='1.0.0',
      description='A Python software toolkit to obtain automatic peak identification for elemental analysis',
      url='https://github.org/specPeakpy/specPeak',
      author='Camille Hebert',
      author_email='camille.hebert.4@ulaval.ca',
      license='MIT License',
      install_requires=['scipy','matplotlib', 'numpy'],
      packages=['specPeak'],
      zip_safe=False,
      package_data={'specPeak':['reference_data/*.dat']},
      include_package_data=True)
