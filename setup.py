from setuptools import setup, find_packages

setup(name='moduleformer',
      packages=find_packages(), 
      install_requires=[
            'torch',
            'transformers'
      ])