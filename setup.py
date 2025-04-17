from pathlib import Path
from setuptools import find_packages, setup, Extension
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize


def get_requirements():
    with open(Path('requirements.txt'), 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_extension_modules():
    """Returns a list of all Cython extension modules"""
    return []


class CustomBuildExt(build_ext):
    def run(self):
        extensions = get_extension_modules()
        ext_modules = cythonize(extensions)
        self.extensions = ext_modules
        build_ext.run(self)


class CustomInstall(install):
    def run(self):
        self.run_command('build_ext')
        install.run(self)


setup(
    name="wdbxpy",
    version="0.1",
    packages=find_packages(),
    ext_modules=cythonize(get_extension_modules()),
    install_requires=get_requirements(),
    setup_requires=['cython'],
    cmdclass={
        'build_ext': CustomBuildExt,
        'install': CustomInstall,
    },
    author="WDBX Team",
    author_email="wdbx@team.com",
    url="https://github.com/WDBX/WDBX",
    description="WDBX Python package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
)
