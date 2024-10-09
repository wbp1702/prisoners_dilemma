import utilities as utils
import numpy as np
import numpy.typing as npt

# Groups split (agents can change tags)
# Split modes in ['empty', 'smallest', 'equal']
def group_split_game(num_agents: int, num_groups: int, num_generations: int, 
                     cost_benefit_ratio: float, cooperation_threshold: float, group_split_size: int, split_mode: str):
    population = [utils.Agent(np.random.randint(num_groups), np.random.randint(2)) for _ in range(num_agents)]
    groups: dict[int, list[utils.Agent]] = { }
    for agent in population:
        if agent.tag in groups:
            groups[agent.tag].append(agent)
        else:
            groups[agent.tag] = [agent]

    recorded_group_population = [[] for _ in range(num_groups)]
    recorded_group_cooperators = [[] for _ in range(num_groups)]
    recorded_group_rewards = [[] for _ in range(num_groups)]

    # Population / Non-empty Groups
    recorded_avg_group_population = []
    recorded_population_cooperators = []
    recorded_population_rewards = []

    recorded_tournament_strategy_changes = [[] for _ in range(4)]
    recorded_group_split_strategy_changes = [[] for _ in range(4)]

    for idx in range(num_groups):
        if idx in groups:
            recorded_group_population[idx].append(len(groups[idx]))
            recorded_group_cooperators[idx].append(np.sum([agent.cooperates() for agent in groups[idx]]))
            recorded_group_rewards[idx].append(np.sum([agent.reward for agent in  groups[idx]]))
        else:
            recorded_group_population[idx].append(0.0)
            recorded_group_cooperators[idx].append(0.0)
            recorded_group_rewards[idx].append(0.0)

    for generation in range(num_generations):

        for changes in recorded_tournament_strategy_changes:
            changes.append(0)

        for changes in recorded_group_split_strategy_changes:
            changes.append(0)

        # Play game in groups
        for group in groups.values():
            if len(group) == 0: continue

            actions = [agent.cooperates() for agent in group]
            payoffs = utils.snowdrift_game(actions, cost_benefit_ratio, cooperation_threshold)
            for idx, agent in enumerate(group): agent.reward = payoffs[actions[idx]]

        # print(f"Agent 0 GenStart Group: {population[0].tag} Reward: {population[0].reward}")

        # Tournament selection
        for idx in range(len(population)):
            agent = population[idx]
            other: utils.Agent = np.random.choice(population)
            # if agents reward is higher do nothing, otherwise replace agent with a copy of the other agent.
            if agent.get_reward() < other.get_reward():
                old_tag = agent.tag
                old_strategy = agent.strategy
                other.copy_to(agent)
                new_tag = agent.tag

                groups[old_tag].remove(agent)
                groups[new_tag].append(agent) # Add copy to other's group
                
                # copy.tag = agent.tag
                # groups[agent.tag].append(copy) # Add copy to agent's group

                recorded_tournament_strategy_changes[agent.strategy * 2 + old_strategy][-1] += 1

        # print(f"Agent 0 After TS Group: {population[0].tag}, Strategy: {population[0].strategy}")

        # split_on = "smallest"
        split_on = split_mode

        # Split Groups
        if split_on == "empty":
            # Choses an empty group or creates a new one if none are empty
            # empty_groups = [idx for idx, group in enumerate(groups) if len(group) == 0]
            empty_groups = []
            for g in range(num_groups):
                if g not in groups or len(groups[g]) == 0:
                    empty_groups.append(g)

            tags = list(groups.keys())
            for tag in tags:
                group = groups[tag]

                group_size = len(group)
                if (np.random.rand() <= group_size / group_split_size):
                    if len(empty_groups) > 0:
                        new_group = empty_groups.pop()
                    else:
                        # groups.append([])
                        num_groups += 1
                        new_group = num_groups - 1
                        recorded_group_population.append([0] * generation)
                        recorded_group_cooperators.append([0] * generation)
                        recorded_group_rewards.append([0] * generation)

                    groups[new_group] = group[group_size//2:] # change to random
                    for agent in groups[new_group]: agent.tag = new_group
                    del group[group_size//2:]
        elif split_on == "smallest":
            # Choses the smallest group, unless it is the smallest then nothing happens
            tags = list(groups.keys())
            for tag in tags:
                key = tag
                group = groups[tag]
                
                group_size = len(group)
                if (np.random.rand() <= group_size / group_split_size):

                    new_group = min(groups.items(), key=lambda x: len(x[1]) + (x[0] == tag))[0]
                    if new_group == key: continue

                    # num_clones = len(groups[new_group])
                    # if num_clones > 0:
                    #     pass
                    for agent in groups[new_group]: agent.strategy = np.random.choice(group).strategy

                    groups[new_group] += group[group_size//2:] # change to random
                    for agent in groups[new_group]: agent.tag = new_group
                    del group[group_size//2:]
        elif split_on == "equal":
            # Choses the smallest group and relocates their populations until they have equal size, 
            # unless it is the smallest then nothing happens
            tags = list(groups.keys())
            for tag in tags:
                key = tag
                group = groups[tag]
                
                group_size = len(group)
                if (np.random.rand() <= group_size / group_split_size):

                    new_group = min(groups.items(), key=lambda x: len(x[1]) + (x[0] == tag))[0]
                    if new_group == key: continue

                    # make the old group remain the larger if the sum of sizes is odd.
                    num_relocations = group_size - ((len(groups[new_group]) + group_size) // 2)
                    
                    for agent in groups[new_group]: 
                        old_strategy = agent.strategy
                        agent.strategy = np.random.choice(group).strategy
                        recorded_group_split_strategy_changes[agent.strategy * 2 + old_strategy][-1] += 1

                    groups[new_group] += group[:num_relocations] # change to random
                    for agent in groups[new_group]: agent.tag = new_group
                    del group[:num_relocations]

        # print(f"Agent 0 After GS Group: {population[0].tag}, Strategy: {population[0].strategy}")

        # Record Statistics
        for idx in range(num_groups):
            if idx in groups:
                recorded_group_population[idx].append(len(groups[idx]))
                recorded_group_cooperators[idx].append(np.sum([agent.cooperates() for agent in groups[idx]]))
                recorded_group_rewards[idx].append(np.sum([agent.reward for agent in  groups[idx]]))
            else:
                recorded_group_population[idx].append(0.0)
                recorded_group_cooperators[idx].append(0.0)
                recorded_group_rewards[idx].append(0.0)

        num_non_empty_groups = 0
        num_cooperators = 0
        num_rewards = 0
        for idx in range(num_groups):
            if recorded_group_population[idx][-1] > 0: num_non_empty_groups += 1
            num_cooperators += recorded_group_cooperators[idx][-1]
            num_rewards += recorded_group_rewards[idx][-1]

        recorded_avg_group_population.append(num_agents / num_non_empty_groups)
        recorded_population_cooperators.append(num_cooperators)
        recorded_population_rewards.append(num_rewards)

    return (recorded_group_population, recorded_group_cooperators, recorded_group_rewards, 
            recorded_avg_group_population, recorded_population_cooperators, recorded_population_rewards,
            recorded_tournament_strategy_changes, recorded_group_split_strategy_changes)

def group_split_game_averaged(num_runs: int, num_agents: int, num_groups: int, num_generations: int, 
                     cost_benefit_ratio: float, cooperation_threshold: float, group_split_size: int, split_mode: str):
    cooperation_ratios = []
    for i in range(num_runs):
        results = group_split_game(num_agents, num_groups, num_generations, 
                                   cost_benefit_ratio, cooperation_threshold, group_split_size, split_mode)
        
        cooperation_ratios.append(results[4][-1] / num_agents)

    return cooperation_ratios
    
def group_split_game_many(num_simulations, *args):
    return [group_split_game(*args) for _ in range(num_simulations)]





# Agents play a n-player game and are reproduced by fitness
def nplayer_group_game(num_agents: int, num_groups: int, num_generations: int, 
                       cost_benefit_ratio: float, mu_tag: float, mu_strategy: float):
    population = [utils.MutatingAgent(np.random.randint(num_groups), np.random.randint(2), mu_tag, mu_strategy) for _ in range(num_agents)]
    groups: dict[int, list[utils.Agent]] = {}
    for agent in population:
        if agent.tag in groups: groups[agent.tag].append(agent)
        else: groups[agent.tag] = [agent]
    
    recorded_group_population = [[] for _ in range(num_groups)]
    recorded_group_cooperators = [[] for _ in range(num_groups)]
    recorded_group_rewards = [[] for _ in range(num_groups)]
    recorded_proportion_cooperators = []
    recorded_num_groups = []
    recorded_avg_reward = []

    for generation in range(num_generations):

        # Record Statistics
        num_non_empty_groups = 0
        # for idx in range(num_groups): 
            # recorded_group_population[idx].append(len(groups[idx]))
            # recorded_group_cooperators[idx].append(np.sum([agent.cooperates() for agent in groups[idx]]))
            # recorded_group_rewards[idx].append(np.sum([agent.reward for agent in  groups[idx]]))
            
        for group in groups.values():
            if len(group) > 0: num_non_empty_groups += 1
        recorded_proportion_cooperators.append(np.sum([np.sum([agent.cooperates() for agent in group]) for group in groups.values()]) / num_agents)
        recorded_num_groups.append(num_non_empty_groups)

        # Play game in groups
        for group_idx, group in groups.items():
            actions = [agent.cooperates() for agent in group]
            payoffs = utils.snowdrift_game(actions, cost_benefit_ratio)
            for agent_idx, agent in enumerate(group): agent.reward = payoffs[actions[agent_idx]]
        recorded_avg_reward.append(np.sum([agent.reward for agent in population]) / num_agents)

        # Tournament selection
        new_population = []
        for agent_idx, agent in enumerate(population):
            other: utils.Agent = np.random.choice(population)
            # if agents reward is higher do nothing, otherwise replace agent with a copy of the other agent.
            if agent.get_reward() < other.get_reward():
                copy = other.copy(num_groups)
                
                if agent not in groups[agent.tag]:
                    pass
                groups[agent.tag].remove(agent)
                
                if copy.tag in groups:
                    groups[copy.tag].append(copy) # Add copy to other's group
                else:
                    groups[copy.tag] = [copy]

                # copy.tag = agent.tag
                # groups[agent.tag].append(copy) # Add copy to agent's group
                
                new_population.append(copy)
            else: new_population.append(agent)
        population = new_population

    return (recorded_group_population, recorded_group_cooperators, recorded_group_rewards,
            recorded_proportion_cooperators, recorded_num_groups, recorded_avg_reward)





class SkilledAgent:
    def __init__(self, tag= None, skill= None):
        self.tag = tag if tag else np.random.rand()
        self.skill = skill if skill else np.random.rand()
        self.rewards = 0

# A game where groups of agents must posses enough agents of each skill to recieve ideal rewards.
# NOTE the number of elements in the 'skill_requirements' parameter determines the number of skills agents will be assigned
# NOTE the sum of skill_requirements should be less than or equal to 1
def skills_game(agent_count: int, group_count: int, generation_count: int, tag_mutation_rate: float,
                skill_mutation_rate: float, skill_reqirements: npt.NDArray, benefit: float) -> None:
    population = [SkilledAgent() for _ in range(agent_count)]
    
    silos = [[] for _ in range(group_count)]
    for agent in population:
        silos[int(np.floor(agent.skill * group_count))].append(agent)

    num_skills = len(skill_reqirements)

    number_of_occupied_groups = np.zeros(generation_count)
    percentage_of_skills_per_group = np.zeros((group_count, num_skills, generation_count))
    rewards_recieved_per_group = np.zeros((group_count, generation_count))


    for gen in range(generation_count):
        for idx, silo in enumerate(silos):
            if not silo: continue
            number_of_occupied_groups[gen] += 1

            skill_count = np.zeros(num_skills)
            for agent in silo: skill_count[int(np.floor(agent.skill * num_skills))] += 1
            # skill_percentages /= len(silo) 
            
            for skill_idx in range(num_skills): percentage_of_skills_per_group[idx, skill_idx, gen] = skill_count[skill_idx] / len(silo)
            
            num_requirements_met = np.sum(skill_count > skill_reqirements)

            # if num_requirements_met == len(skill_reqirements):
            skill_rewards = [(benefit / len(skill_count)) / count for count in skill_count]
            
            for agent in silo: agent.rewards += skill_rewards[int(np.floor(agent.skill * num_skills))]
            
        for agent in population:
            pass