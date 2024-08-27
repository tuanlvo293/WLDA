from setuptools import setup, find_packages

setup(
    name='WLDA',
    version='0.1.0',
    description='My awesome WLDA package',
    url='https://github.com/tuanlvo293/WLDA.git',
    author='Tuan L. Vo',
    author_email='tuanlvo293@gmail.com',
    package_dir={"": "src"},  # Chỉ định thư mục gốc của các packages là "src"
    packages=find_packages(where="src"),  # Tìm các packages trong thư mục "src"
    keywords='python package',
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
)
