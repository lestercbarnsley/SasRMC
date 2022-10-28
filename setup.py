
'''setup.py
module for autogenerating pakage metadata
'''


from setuptools import setup, find_packages

with open('README.md') as rm:
    LONG_DESCRIPTION = rm.read()

setup(
    name='sas_rmc',
    version='0.0.4',
    author='Lester Barnsley',
    author_email='lester.barnsley@gmail.com',
    long_description=LONG_DESCRIPTION,
    url='', # the url of your packge, can be pypi
    licence="MIT",
    packages=find_packages(include=["sas_rmc", "sas_rmc.*"]),#["sas_rmc"],  # packages=setuptools.find_packages(),
    #package_dir={'sas_rmc': "./sas_rmc"},
    install_requires=[
        'PyYAML',
        'pandas',
        'numpy',
        'scipy',
        'matplotlib',
        'openpyxl',
        'ipython'
        ],
)

