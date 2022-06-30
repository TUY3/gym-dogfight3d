# DogFight3d: a 3d air combat reinforcement learning environment

DogFight3d is a reinforcement learning environment for training fixed-wing aircraft 1v1 air combat, it has approximate 
real dynamics.

## Dependencies
* gym, numpy, transform3d
* python3.8
* linux or win

## Installation
gym-dogfight3d is pip installable using its GitHub:

```
pip install git+https://github.com/TUY3/gym-dogfight3d
```
or
```angular2html
git clone https://github.com/TUY3/gym-dogfight3d
cd gym-dogfight3d
pip install -e .
```

## Environment
### observation space
    # ownship
    position(x,y,z)
    linear speed
    linear acceleration
    health level
    cap(yaw),pitch,roll
    thrust_level
    # opponent
    position(x,y,z)
    linear speed
    cap(yaw),pitch,roll
### action space
continuous action space, including throttle,elevator,aileron,rudder

## Example
```
env = gym.make('Dogfight3d-v0')
obs_space = env.observation_space
action_space = env.action_space
init_obs = env.reset()
while True:
    action = env.action_space.sample()
    next_obs, r, done, info = env.step(action)
    if done:
        print(info, env.current_step)
        break
```

### Reference
* harfang3d/dogfight-sandbox-hg1

