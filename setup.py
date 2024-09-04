from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='prov4ml',
    version='1.0.0',
    packages=find_packages(),
    install_requires=required,  # Loaded from requirements.txt
    extras_require={
        'apple': [
            # Optional dependencies for Apple/Mac
            'apple_gpu==0.3.*'
        ],
        'linux': [
            # Optional dependencies for Linux
            'pyamdgpuinfo==2.1.*'
        ]
    }
)
