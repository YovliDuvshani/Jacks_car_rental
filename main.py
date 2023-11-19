from env import Env
from agent import Agent

Env().transitions(1)

agent = Agent(Env())
agent.improve_policy()
