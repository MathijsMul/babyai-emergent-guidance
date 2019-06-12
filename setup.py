from setuptools import setup

setup(
    name='babyai',
    version='0.0.2',
    license='BSD 3-clause',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    packages=['babyai', 'babyai.levels', 'babyai.utils'],
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.10.0',
        'pyqt5>=5.10.1',
        'torch==1.0.0',
        'gym_minigrid @ git+https://github.com/maximecb/gym-minigrid.git/@dba71a2c63be34ee061035c60e857197466c87db#egg=gym_minigrid',
        'blosc>=1.5.1',
        'sklearn'
    ]
)
