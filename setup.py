from setuptools import setup, find_packages

MODULE_NAME = 'datarisk_mlops_codex'
MODULE_NAME_IMPORT = 'mlops_codex'
REPO_NAME = 'mlops_codex'
MODULE_VERSION='2.2.7'


def requirements_from_pip(filename='requirements.txt'):
    with open(filename, 'r') as pip:
        return [l.strip() for l in pip if not l.startswith('#') and l.strip()]


setup(name=MODULE_NAME,
      description="Python tools for interact with Datarisk MLOps platform",
      url='https://datarisk-io.github.io/mlops_codex/',
      author="Datarisk",
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      version=MODULE_VERSION,
      download_url=f'https://github.com/datarisk-io/mlops_codex/archive/refs/tags/v{MODULE_VERSION}.tar.gz',
      install_requires=requirements_from_pip(),
      include_package_data=True,
      zip_safe=False,
      classifiers=['Programming Language :: Python :: 3'])
