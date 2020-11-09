from setuptools import setup, find_packages

setup(name='ccmgp',
      description='Modeling the Music Genre Perception across Language-Bound Cultures',
      author='Deezer Research',
      install_requires=['numpy==1.17.2',
                        'pandas==0.25.1',
                        'sklearn==0.0',
                        'networkx==2.2',
                        'joblib==0.13.2',
                        'torch==1.4.0',
                        'SPARQLWrapper==1.8.4',
                        'mecab-python3==0.996.5',
                        'transformers==2.4.1'],
      package_data={'ccmgp': ['README.md']},
      packages=find_packages())
