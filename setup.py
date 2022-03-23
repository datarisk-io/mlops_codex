from setuptools import setup

setup(
    name='Neomaril Logger',
    version='1.0',
    install_requires = [
        'hy==1.0a1',
        'funcparserlib==1.0.0a0'
    ],
    packages=['.neomaril_logger'],
)
