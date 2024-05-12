from setuptools import setup, find_packages

setup(
    name='missing_mga',
    version='0.1',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'matplotlib', 'seaborn', 'itertools', 'upsetplot'],
    author='Mariano Gobea Alcoba',
    author_email='gobeamariano@gmail.com',
    description='A package for handling missing values in datasets.',
    url='https://github.com/Mgobeaalcoba/missing_mga.git',
    license='MIT',
)
