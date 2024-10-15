from setuptools import find_packages,setup
#this for finding the packages that we used on our app

setup(

    name = 'mlproject',
    version='0.0.1',
    author='keddis',
    author_email='mouadkeddis430@gmail.com',
    packages=find_packages(),
    install_requires= ['pandas','numpy','seaborn'],

)