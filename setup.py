from setuptools import setup

setup(
    name='WLDA',
    version='0.1.0',
    description='My awesome WLDA package',
    url='https://github.com/tuanlvo293/WLDA.git',
    author='Tuan L. Vo',
    author_email='tuanlvo293@gmail.com',
    package_dir={'': 'src'},  # Chỉ định thư mục chứa module là "src",
    keywords='python package',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
)
