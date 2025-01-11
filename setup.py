from setuptools import setup, find_packages

setup(
    name='rllm',
    version='0.0.0',
    description='Distributed Post-Training RL Library for LLMs',
    author='RLLM Team',
    packages=find_packages(include=['rllm',]),
    install_requires=[
        # List your package dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)