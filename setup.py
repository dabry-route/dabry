from setuptools import setup, find_packages

setup(
    name='dabry',
    version='1.1',
    description='Trajectory optimization in flow fields',
    author='Bastien Schnitzler',
    author_email='bastien.schnitzler@live.fr',
    packages=find_packages('.'),
    package_dir={'': '.'},
    install_requires=['h5py', 'numpy', 'scipy', 'tqdm']
)
