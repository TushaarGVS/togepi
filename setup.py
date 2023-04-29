from setuptools import setup, find_packages

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setup(name='togepi',
      version="1.0",
      description='toeplitz-based generative pretraining',
      author='Tushaar Gangavarapu',
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=requirements,
      url='https://github.com/TushaarGVS/togepi')
