from os.path import join

from setuptools import setup, find_packages

MODULE_NAME = 'neomaril_codex'
MODULE_NAME_IMPORT = 'neomaril_codex'
REPO_NAME = 'mlops-neomaril-codex'
MODULE_VERSION='0.1'

def requirements_from_pip(filename='requirements.txt'):
    with open(filename, 'r') as pip:
        return [l.strip() for l in pip if not l.startswith('#') and l.strip()]


setup(name=MODULE_NAME,
      description="Python tools for interact with Neomaril",
      url='https://github.com/datarisk-io/{:s}'.format(REPO_NAME),
      author="Datarisk",
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version=MODULE_VERSION,
      install_requires=requirements_from_pip(),
      include_package_data=True,
      zip_safe=False,
      classifiers=['Programming Language :: Python :: 3.9'])
