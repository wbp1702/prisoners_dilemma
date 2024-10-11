import numpy as np
import matplotlib.pylab as plt

class Agent:
    def __init__(self, tag: int, strategy: int):
        self.strategy = strategy
        self.tag = tag
        self.reward = 0

    def copy(self, num_tags):
        return Agent(self.tag, self.strategy)
    
    def copy_to(self, target):
        target.strategy = self.strategy
        target.tag = self.tag
    
    def get_reward(self) -> float:
        return self.reward
    
    def add_reward(self, reward):
        self.reward += reward
    
    def cooperates(self) -> bool:
        return bool(self.strategy)

class MutatingAgent(Agent):
    def __init__(self, tag: int, strategy: int, mu_tag: float, mu_strategy: float):
        if not mu_strategy or not mu_tag or mu_strategy < 0 or mu_strategy > 1 or mu_tag < 0 or mu_tag > 1: 
            raise ValueError("Strategy and Tag Mutation Rate must be in the interval [0, 1]")
        
        Agent.__init__(self, tag, strategy)
        self.mu_tag = mu_tag
        self.mu_strategy = mu_strategy

    def copy(self, num_tags):
        return MutatingAgent(self.tag if np.random.rand() > self.mu_tag else np.random.randint(num_tags), 
                             self.strategy if np.random.rand() > self.mu_strategy else np.random.randint(2),
                             self.mu_tag,
                             self.mu_strategy)

# Generates a new population using Tournament Selection (k=2, p=1)
def tournament_selection(population: list[Agent]):
    new_population = []

    for agent in population:
        other: Agent = np.random.choice(population)
        if agent.get_reward() > other.get_reward():
            new_population.append(agent.copy())
        else:
            new_population.append(other.copy())

    return new_population

# Generates a new population using Roulette Selection
def roulette_selection(population: list[Agent]):
    new_population = []

    rewards = np.array([agent.get_reward() for agent in population])
    probabilities = rewards / np.sum(rewards)

    for i in range(len(population)):
        choice: Agent = np.random.choice(population, p=probabilities)
        new_population.append(choice.copy())


# Calculate the payoffs for a N-Player Social Dilemma 
def social_dilemma(actions: list[bool], cost_benefit_ratio: float, cooperation_threshold: float, normalize_rewards: bool= True) -> tuple[float, float]:
    if cost_benefit_ratio <= 0: raise ValueError("cost_benefit_ratio must be greater than zero")
    if cooperation_threshold < 0 or cooperation_threshold > 1: raise ValueError("cooperation_threshold must be greater inside the interval [0, 1]")

    benefit = 1 / cost_benefit_ratio
    cost = 1
    
    num_cooperators = np.sum(actions)

    if cooperation_threshold and num_cooperators < np.ceil(cooperation_threshold * len(actions)):
        payoffs = np.array([
            0, # defectors
            -cost * (len(actions) - 1) / np.ceil(cooperation_threshold * len(actions)) # cooperators
        ])
    else:
        payoffs = np.array([
            benefit * num_cooperators, # decectors
            benefit * num_cooperators - cost * (len(actions) - 1) / (num_cooperators + 1) # cooperators
        ])

    if normalize_rewards:
        payoffs = (payoffs - np.min(payoffs)) / (np.max(payoffs) - np.min(payoffs))

    return (payoffs[0], payoffs[1])

# Calculate the payoffs for a N-Player Snowdrift Game 
def snowdrift_game(actions: list[bool], cost_benefit_ratio: float, cooperation_threshold: float) -> tuple[float, float]:
    if cost_benefit_ratio <= 0: raise ValueError("cost_benefit_ratio must be greater than zero")
    
    # benefit = 1 / cost_benefit_ratio
    # cost = 1
    benefit = 1
    cost = cost_benefit_ratio
    
    num_cooperators = np.sum(actions)

    # payoffs = np.array([
    #     benefit if num_cooperators > 0 else 0, # decectors
    #     (benefit - cost / num_cooperators) if num_cooperators else 0 # cooperators
    # ])

    # return (payoffs[0], payoffs[1])
    
    # num_cooperators = np.sum(actions)
    # benefit = 1 / cost_benefit_ratio
    # cost = 1

    if cooperation_threshold == 0 and num_cooperators == 0:
        defector_payoff = benefit
        cooperator_payoff = benefit
    # elif num_cooperators / len(actions) >= cooperation_threshold:
    elif num_cooperators >= cooperation_threshold:
        defector_payoff = benefit
        cooperator_payoff = benefit - cost / num_cooperators
    else:
        defector_payoff = 0
        cooperator_payoff = 0
        
    return (defector_payoff, cooperator_payoff)

# Calculate the payoffs for a N-Player Prisoner's Dilemma 
def prisoners_dilemma(actions: list[bool], cost_benefit_ratio: float) -> tuple[float, float]:
    if cost_benefit_ratio <= 0: raise ValueError("cost_benefit_ratio must be greater than zero")
    
    benefit = 1 / cost_benefit_ratio
    cost = 1
    
    num_cooperators = np.sum(actions)

    payoffs = np.array([
        benefit * num_cooperators, # decectors
        benefit * num_cooperators - cost * (len(actions) - 1) # cooperators
    ])

    return (payoffs[0], payoffs[1])
