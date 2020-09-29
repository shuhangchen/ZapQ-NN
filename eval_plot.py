import math
import os
import random
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# specify the path where rewards are stored.
r_directory = './sdata/rewards/' 

data_r = []

for filename in os.listdir(r_directory):
    if filename.endswith('.pt'):
        r = torch.load(r_directory + filename)
        data_r.append(r)

print 'rewards collection down. ' + str(len(data_r))

xaxis_episode = [(i+1)*200 for i in range(len(data_r[0]))]

rewards = data_r


def stat_rewards(rewards, stat):
    return [stat([r[i] for r in rewards]) for i in range(len(xaxis_episode))]

def sigma(l):
    mean = sum(l)/float(len(l))
    return math.sqrt(sum([(r - mean)**2 for r in l]) / (len(l) - 1))

rewards_mean = stat_rewards(rewards, lambda l: sum(l)/float(len(l)))
rewards_sigma = stat_rewards(rewards, sigma)
rewards_up = [rewards_mean[i] + rewards_sigma[i] for i in range(len(rewards_mean))]
rewards_down = [rewards_mean[i] - rewards_sigma[i] for i in range(len(rewards_mean))]
rewards_final = [r[-1] for r in rewards]

rewards_median = stat_rewards(rewards, lambda l: np.percentile(l, 50))
rewards_90 = stat_rewards(rewards, lambda l: np.percentile(l, 90))
rewards_75 = stat_rewards(rewards, lambda l: np.percentile(l, 75))
rewards_25 = stat_rewards(rewards, lambda l: np.percentile(l, 25))
rewards_10 = stat_rewards(rewards, lambda l: np.percentile(l, 10))

###########################################################
fig = plt.figure(1, figsize = (6, 4))


ax = fig.add_subplot(1, 1, 1)
ax.plot(xaxis_episode, rewards_75, color = 'blue', alpha=0.2)
ax.plot(xaxis_episode, rewards_median, color = 'blue', linestyle ='-.', linewidth = 2, label=r'median')
ax.plot(xaxis_episode, rewards_25, color = 'blue', alpha=0.2)
ax.fill_between(xaxis_episode, rewards_10, rewards_25, color = 'blue', alpha=0.2, label='10-25')
ax.fill_between(xaxis_episode, rewards_25, rewards_75, color = 'blue', alpha=0.4, label='25-75')
ax.fill_between(xaxis_episode, rewards_75, rewards_90, color = 'blue', alpha=0.6, label='75-90')
ax.legend(loc='lower right')
ax.set_ylim([0, 1000])
ax.set_xlabel('Episodes')
ax.set_ylabel(r'$\mathcal{R}(\phi^{\theta})$')


fig.savefig('./p_pole_20-ds-eps.svg', format='svg', dpi=1200)
fig.savefig('./p_pole_20-ds-eps.pdf', format='pdf')