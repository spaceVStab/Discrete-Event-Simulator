# Discrete-Event-Simulator
Simulator engine for integrating your Reinforcement Learning model for scheduling some set of processes across some set of resources. This is the outcome of my B.Tech project to apply RL models for scheduling factory jobs across machines. These codes suffices just for the basics required for a proto-type RL model.

`Files`
- `model.py` : Provides class definition for Machines and Jobs. These classes are responsible for maintaining status records for each machines and jobs which are initiated.
- `environment.py` : Initiate the environment class with `config` parameter dictionary. Provides functionality of `reset()`, `step()` and `possibleAction()`. Since the action space is not fixed and thus for every state observation, set of action is different. `possibleAction()` returns the set of action for a state observation.

`Usage`
- Define your `config` parameter dictionary and pass it as an argument to environment class. See `QLearner.py` for reference. 

`Added Features`
- For fixed number of jobs and fixed number of machines, with ordering constraint among the jobs
- Naive Q learning algorithm 
- Deep Q Network added but many issues training it

`TODO`
- Provide an interface for gym
- Add functionality for continuous pool of jobs with fix number of machines
- Fix issues around DQN 
