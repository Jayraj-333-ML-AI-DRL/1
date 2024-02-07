
from setuptools import find_packages,setup
#from typing import List 


Hypen_dot_E = "-e ."


def get_requirements(filepath:str):
        """Returns a list of requirements from the given file path."""  
        requirements = []
        with open(file=filepath,mode='r') as f:
            requirements = f.readlines()
            requirements = [req.replace("\n","") for req in requirements]
            
        if Hypen_dot_E in  requirements:
            #             print("Found -e option.")
            requirements = requirements.remove(Hypen_dot_E)
        
        return requirements
     
setup(name='Project 2',
    version='1.0',
    #description='Python Distribution Utilities',
    author='Jayraj Rajput',
    author_email='jayrajput1997@gmail.com',
    packages=find_packages(),
    install_requirements = get_requirements('requirements.txt'))

