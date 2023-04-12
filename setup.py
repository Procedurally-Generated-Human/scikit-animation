from setuptools import find_packages, setup
setup(
    name='scikit-animation',
    packages=find_packages(include=['scikit_animation']),
    version='0.1.0',
    description='Library for animating scikit-learn models',
    author='Parsa Toopchinezhad',
    license='MIT',
    install_requires=['matplotlib', 'numpy', 'scikit-learn', 'celluloid'],
)