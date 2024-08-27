from setuptools import setup, find_packages

setup(
    name='WLDA',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    description='A Python library for Weighted Linear Discriminant Analysis (WLDA)',
    author='Tuan L. Vo',
    author_email='tuanlvo293@gmail.com',
    url='https://github.com/tuanlvo293/WLDA.git'',
    license='MIT',
)
