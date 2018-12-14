# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='metrics',
    version='1.0.0',
    description='Package for evaluete clusters',
    long_description=readme,
    author='Weslley Caldas',
    author_email='weslleylc@gmail.com',
    url='https://github.com/weslleylc/metrics',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

