import gymnasium as gym
import numpy as np
import statistics

class Qlearner():
    def __init__(self,
                gymEnv: gym.Env,
                episodes: int,
                beta: float,
                omega: float,
                ):
        self.env = gymEnv
        self.episodes = episodes
        self.beta = beta
        self.omega = omega
        self.gainTable = np.zeros((gymEnv.observation_space.n, gymEnv.action_space.n))


    def learn(self, actionSelector: callable, testPeriod: int, numberOfTests):
        currEpisode = 0
        while currEpisode < self.episodes:
            state, info = self.env.reset()
            terminated, truncated = False, False
            if currEpisode % testPeriod == 0:
                median, avg = self.evaluate(numberOfTests, actionSelector)
                print(f"avg: {avg:02f},  median: {median:02f}   episode: {currEpisode}")
            while not terminated and not truncated:
                "truncated after 200 moves"
                action = actionSelector(state, self.gainTable, info['action_mask'], self.env.action_space.n, False)
                new_state, tempReward, terminated, truncated, info = self.env.step(action)

                delta = tempReward + self.omega * max(self.gainTable[new_state]) - self.gainTable[state][action]
                self.gainTable[state][action] += self.beta * delta

                state = new_state
            currEpisode += 1

    def evaluate(self, test_count: int, actionSelector: callable):
        reward_sum = []
        for i in range(test_count):
            state, info = self.env.reset()
            terminated, truncated = False, False
            tempReward = 0
            while not terminated and not truncated:
                action = actionSelector(state, self.gainTable, info['action_mask'], self.env.action_space.n, True)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                tempReward += reward
                state = new_state
            reward_sum.append(tempReward)
        return statistics.median(reward_sum), sum(reward_sum) / test_count


def BoltzmannSelection(state: int, gainTable: np.ndarray, action_mask, actionNumber: int, evaluate: bool):
        action_prob = np.exp(gainTable[state] / 1) * action_mask
        prob_sum = sum(action_prob)
        prob = action_prob / prob_sum
        if evaluate is False:
            return np.random.choice(np.arange(actionNumber), p=prob)
        else:
            idxOfBest = [i for i in range(0, len(action_prob)) if action_prob[i] == max(action_prob)]
            return np.random.choice(idxOfBest)

def main(problem):
    maxEpisodes = 10000
    beta = 0.15
    omega = 0.9
    numberOfTests = 30
    testAfter = 500
    selector = BoltzmannSelection
    env = gym.make(problem)
    alg = Qlearner(env, maxEpisodes, beta, omega)
    alg.learn(selector, testAfter, numberOfTests)
    median, avg = alg.evaluate(numberOfTests, selector)
    print(f"avg: {avg:02},  median: {median:02}   episode: {maxEpisodes}")


if __name__ == '__main__':
    main('Taxi-v3')