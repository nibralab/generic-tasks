from setuptools import setup

setup(
    name='generic_tasks',
    version='0.1.0',
    description='A set of generic tasks for AI prototypes',
    url='https://github.com/nibralab/generic_tasks',
    author='Niels Braczek',
    author_email='nbraczek@bsds.de',
    license='MIT',
    packages=['generic_tasks'],
    install_requires=[
        'torch',
        'transformers',
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
