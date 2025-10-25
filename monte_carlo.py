import gymnasium as gym
from gymnasium.envs.registration import register
import random
import gym_tictactoe

register(
    id="TicTacToe-v0",
    entry_point="gym_tictactoe.envs.tictactoe_env:TicTacToeEnv",
)

env = gym.make("TicTacToe-v0")
obs, info = env.reset()
env.render()

done = False

while not done:
    # Available actions for current player
    available_actions = [i for i, x in enumerate(obs.flatten()) if x == 0]
    action = random.choice(available_actions)

    # Take action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    done = terminated or truncated

print("Game over! Reward:", reward)
