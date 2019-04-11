# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='investment-machine_predict-prices',
    version='0.0.1',
    description='Predict stock prices',
    long_description=readme,
    author='u6k',
    author_email='u6k.apps@gmail.com',
    url='https://github.com/u6k/investment-machine_predict-prices',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

