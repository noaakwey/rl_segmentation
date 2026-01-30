"""
Setup script for Crop Segmentation (Supervised U-Net) package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='crop_segmentation',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Supervised U-Net Crop Segmentation from Satellite Imagery',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/crop_segmentation',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: GIS',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'black>=23.0',
            'flake8>=6.0',
            'ipython>=8.0',
            'jupyter>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'unet-train=src.supervised_train:main',
            'unet-eval=src.supervised_eval:main',
            'unet-infer=src.production_unet_infer:main',
        ],
    },
    include_package_data=True,
    package_data={
        'configs': ['*.yaml'],
    },
)
