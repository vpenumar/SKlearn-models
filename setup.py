
from setuptools import setup, find_packages

setup(
    name='hunga_bunga',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='Brute-Force All of sklearn!',
    long_description=open('README.txt').read(),
    install_requires=['numpy'],
    url='https://github.com/ypeleg/HungaBunga',
    author='Yam Peleg',
    author_email='yam@deeptrading.com'
)
