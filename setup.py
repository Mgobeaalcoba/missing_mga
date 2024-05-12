from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='missing_mga',
    version='0.4.3',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'matplotlib', 'seaborn', 'upsetplot', 'scikit-learn'],
    author='Mariano Gobea Alcoba',
    author_email='gobeamariano@gmail.com',
    description='A package for handling missing values in datasets.',
    long_description=long_description,  # Usa el contenido del README.md como descripción larga
    long_description_content_type="text/markdown",  # Especifica el tipo de contenido como markdown
    url='https://github.com/Mgobeaalcoba/missing_mga',
    license='MIT',
)
