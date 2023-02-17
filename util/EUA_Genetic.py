import random
from util.utils import get_reward, mask_trans_to_list
import matplotlib.pyplot as plt
import tqdm


chromosome_num = 100
# 交叉概率
pC = 0.5
# 变异概率
pM = 0.05
# 迭代轮数
iter_count = 2000


class Chromosome:
    def __init__(self, user_within_servers=None, genes=None):
        if genes is None:
            genes = [random.choice(user_within_servers[i]) for i in range(len(user_within_servers))]
        self.genes = genes
        self.reward = 0
        self.user_allocated_prop = 0
        self.server_used_prop = 0
        self.capacity_used_prop = 0
        self.user_allocate_list = None
        self.server_allocate_num = None

    def calc_reward(self, servers, users):
        self.user_allocate_list, self.server_allocate_num, \
            self.user_allocated_prop, self.server_used_prop, self.capacity_used_prop \
            = get_reward(servers, users, self.genes)
        self.reward = 0.5 * self.user_allocated_prop + 0.5 * self.capacity_used_prop

    def __lt__(self, other):
        return self.reward < other.reward

    def __eq__(self, other):
        return self.reward == other.reward

    def __gt__(self, other):
        return self.reward > other.reward


def genetic_allocate(servers, users, user_masks):
    global pC, pM
    server_num = len(servers)
    user_within_servers = mask_trans_to_list(user_masks, server_num)
    rewards = []
    chromosomes = [Chromosome(user_within_servers) for _ in range(chromosome_num)]
    for chromosome in chromosomes:
        chromosome.calc_reward(servers, users)
    best_chromosomes = max(chromosomes)
    rewards.append(best_chromosomes.reward)
    generation_count = 0
    # t_bar = tqdm.tqdm(total=iter_count)  # 初始化进度条
    for _ in range(iter_count):
        # t_bar.update(1)  # 更新进度
        # t_bar.set_description("{:2f} {:2f} {:2f} {:2f}".format(best_chromosomes.reward,
        #                                                        best_chromosomes.user_allocated_prop,
        #                                                        best_chromosomes.server_used_prop,
        #                                                        best_chromosomes.capacity_used_prop))  # 更新描述
        # t_bar.refresh()  # 立即显示进度条更新结果
        chromosomes = generate_next_generation(chromosomes, user_within_servers, best_chromosomes)
        generation_count += 1
        for chromosome in chromosomes:
            chromosome.calc_reward(servers, users)
        best_chromosomes = max(chromosomes)
        rewards.append(best_chromosomes.reward)
        pC += (1 - 0.5) / iter_count
        pM += (1 - 0.05) / iter_count

    plt.plot(rewards)
    plt.title('reward')
    plt.show()

    return None, None, best_chromosomes.user_allocate_list, best_chromosomes.server_allocate_num, \
        best_chromosomes.user_allocated_prop, best_chromosomes.server_used_prop, best_chromosomes.capacity_used_prop


def choose_one(chromosome_list):
    """
    选择一个染色体，轮盘赌
    """
    reward_list = [chromosome.reward for chromosome in chromosome_list]
    sum_reward = sum(reward_list)
    rand_num = random.uniform(0, sum_reward)
    for i in range(len(chromosome_list)):
        rand_num -= reward_list[i]
        if rand_num <= 0:
            return chromosome_list[i]


def cross(parent1, parent2):
    """
    交叉，把第2个抽出一段基因，放到第1个的相应位置
    """
    n = len(parent1.genes)
    start = random.randint(0, n - 1)
    end = random.randint(start, n - 1)
    new_genes = parent1.genes[:]
    new_genes[start:end] = parent2.genes[start:end]
    return new_genes


def mutate(genes, user_within_servers):
    # 随便改一个
    n = len(genes)
    index = random.randint(0, n - 1)
    genes[index] = random.choice(user_within_servers[index])
    return genes


def generate_next_generation(chromosome_list, user_within_servers, best_chromosome):
    # 把这一代最好的留下来
    new_list = [best_chromosome]
    for i in range(chromosome_num - 1):
        new_c = new_child(chromosome_list, user_within_servers)
        new_list.append(new_c)
    return new_list


def new_child(chromosome_list, user_within_servers):
    parent1 = choose_one(chromosome_list)
    # 决定是否交叉
    rate = random.random()
    if rate < pC:
        parent2 = choose_one(chromosome_list)
        new_genes = cross(parent1, parent2)
    else:
        new_genes = parent1.genes[:]
    # 决定是否突变
    rate = random.random()
    if rate < pM:
        new_genes = mutate(new_genes, user_within_servers)
    new_chromosome = Chromosome(genes=new_genes)
    return new_chromosome
