from setuptools import setup, find_packages

setup(
    name='gym_tictactoe',
    version='0.0.1',
    packages=find_packages(),           # Include all packages under gym_tictactoe
    install_requires=['gymnasium'],     # Use Gymnasium instead of Gym
)