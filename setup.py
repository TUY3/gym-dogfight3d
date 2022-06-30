from setuptools import setup, find_packages


setup(
	name='gym_dogfight3d',
	version='1.0.1',
	packages=find_packages(),
	install_requires=['gym',
					  'transforms3d'
	],
	description='a 3d air combat reinforcement learning environment',
)
