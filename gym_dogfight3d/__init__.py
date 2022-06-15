from gym.envs.registration import register


register(
		id='Dogfight3d-v0',
		entry_point='gym_dogfight3d.envs:DogFightEnv',
)
