"""An example DQN Agent.

It uses pytorch 1.5+ and tensorboard libraries (HINT: these dependencies can
be installed by running pip install nasim[dqn])

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim/agents dir:

$ python dqn_agent.py tiny

To see detailed results using tensorboard:

$ tensorboard --logdir runs/

To see available hyperparameters:

$ python dqn_agent.py --help

Notes
-----

This is by no means a state of the art implementation of DQN, but is designed
to be an example implementation that can be used as a reference for building
your own agents.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import numpy as np
import math
from gym import error
from pprint import pprint
from SumTree2 import SumTree
import nasim
from others import save_data
from torch.autograd import Variable
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install dqn_agent dependencies by running "
        "'pip install nasim[dqn]'.)"
    )


class ReplayMemory:

    def __init__(self, capacity, s_dims, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [self.s_buf[sample_idxs],
                 self.a_buf[sample_idxs],
                 self.next_s_buf[sample_idxs],
                 self.r_buf[sample_idxs],
                 self.done_buf[sample_idxs]]
        return [torch.from_numpy(buf).to(self.device) for buf in batch]

class ReplayMemoryPER:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    
    def __init__(self, capacity,device="cuda"):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.device = device
    def _get_priority(self, error):
        return (abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
    
    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()


        
        batchs = np.array(batch).transpose()
    
        s=torch.from_numpy(np.vstack(batchs[0])).to(self.device)
        h=torch.from_numpy(np.vstack(list(batchs[1])).astype(np.int64)).to(self.device)
        a=torch.from_numpy(np.vstack(list(batchs[2])).astype(np.int64)).to(self.device)     
        r=torch.from_numpy(np.array(list(batchs[3]))).to(self.device)
        s_=torch.from_numpy(np.vstack(batchs[4])).to(self.device)
        d=torch.from_numpy(np.array(list(batchs[5])).astype(np.int32)).to(self.device)
        return s,h,a,s_,r,d, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
class DQN_dueling_noisy(nn.Module):
    """A simple Deep Q-Network """

    def __init__(self, input_dim, layers, action_num,host_num):
        super().__init__()
        self.action_num=action_num
        self.host_num=host_num
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l-1], layers[l]))


        # self.host_layer1 = nn.Linear(layers[-1], layers[-1])
        # self.host_layer_out= nn.Linear(layers[-1], host_num)
        # self.action_layer1 = nn.Linear(layers[-1], layers[-1])
        # self.action_layer_out= nn.Linear(layers[-1], action_num)

        self.host_advantage_hidden_layer = NoisyLinear(layers[-1], layers[-1])
        self.host_advantage_layer = NoisyLinear(layers[-1], host_num)
        self.host_value_hidden_layer = NoisyLinear(layers[-1], layers[-1])
        self.host_value_layer = NoisyLinear(layers[-1], 1)

    
        self.action_advantage_hidden_layer = NoisyLinear(layers[-1], layers[-1])
        self.action_advantage_layer = NoisyLinear(layers[-1], action_num)
        self.action_value_hidden_layer = NoisyLinear(layers[-1], layers[-1])
        self.action_value_layer = NoisyLinear(layers[-1], 1)

    def host_forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        h = F.relu(self.host_layer1(x))
        host = self.host_layer_out(h)
        return host
    def action_forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        a = F.relu(self.action_layer1(x))
        action = self.action_layer_out(a)
        return action

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        host_values = F.relu(self.host_value_hidden_layer(x))
        host_values = self.host_value_layer(host_values)

        host_advantages = F.relu(self.host_advantage_hidden_layer(x))
        host_advantages = self.host_advantage_layer(host_advantages)
        host_qvals = host_values + (host_advantages - host_advantages.mean())


        action_values = F.relu(self.action_value_hidden_layer(x))
        action_values = self.action_value_layer(action_values)

        action_advantages = F.relu(self.action_advantage_hidden_layer(x))
        action_advantages = self.action_advantage_layer(action_advantages)
        action_qvals = action_values + (action_advantages - action_advantages.mean())


        # h = F.relu(self.host_layer1(x))
        # #host = F.relu(self.host_layer_out(h))
        # host = self.host_layer_out(h)
        # a = F.relu(self.action_layer1(x))
        # #action = F.relu(self.action_layer_out(a))
        
        # action = self.action_layer_out(a)
       
        return host_qvals,action_qvals  
    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))


    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            host,action=self.forward(x)
            a=action.max(1)[1]
            h=host.max(1)[1]
            return h,a
    def reset_noise(self):
        self.host_advantage_hidden_layer.reset_noise()
        self.host_advantage_layer.reset_noise()
        self.host_value_hidden_layer.reset_noise()
        self.host_value_layer.reset_noise()

        self.action_advantage_hidden_layer.reset_noise()
        self.action_advantage_layer.reset_noise()
        self.action_value_hidden_layer.reset_noise()
        self.action_value_layer.reset_noise()

class DQN(nn.Module):
    """A simple Deep Q-Network """

    def __init__(self, input_dim, layers, action_num,host_num):
        super().__init__()
        self.action_num=action_num
        self.host_num=host_num
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l-1], layers[l]))


        self.host_layer1 = NoisyLinear(layers[-1], layers[-1])
        self.host_layer_out= NoisyLinear(layers[-1], host_num)
        # self.advantage_hidden_layer = nn.Linear(layers[-1], layers[-1])
        # self.advantage_layer = nn.Linear(layers[-1], num_actions)
        # self.value_hidden_layer = nn.Linear(layers[-1], layers[-1])
        # self.value_layer = nn.Linear(layers[-1], 1)

        self.action_layer1 = NoisyLinear(layers[-1], layers[-1])
        self.action_layer_out= NoisyLinear(layers[-1], action_num)
        # self.advantage_hidden_layer = nn.Linear(layers[-1], layers[-1])
        # self.advantage_layer = nn.Linear(layers[-1], num_actions)
        # self.value_hidden_layer = nn.Linear(layers[-1], layers[-1])
        # self.value_layer = nn.Linear(layers[-1], 1)
    def reset_noise(self):
        self.host_layer1.reset_noise()
        self.host_layer_out.reset_noise()
        self.action_layer1.reset_noise()
        self.action_layer_out.reset_noise()
        #self.value_layer.reset_noise()
    def host_forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        h = F.relu(self.host_layer1(x))
        host = self.host_layer_out(h)
        return host
    def action_forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        a = F.relu(self.action_layer1(x))
        action = self.action_layer_out(a)
        return action

    # def forward(self, x):
    #     for layer in self.layers:
    #         x = F.relu(layer(x))
    #     h = F.relu(self.host_layer1(x))
    #     #host = F.relu(self.host_layer_out(h))
    #     host = self.host_layer_out(h)
    #     a = F.relu(self.action_layer1(x))
    #     #action = F.relu(self.action_layer_out(a))
        
    #     action = self.action_layer_out(a)
    #    # host=host.squeeze()
    #     #action=action.squeeze()
    #     #host1=host.unsqueeze(1)
    #     #host2=host1.expand(-1,self.action_num)
    #     host3=host.expand(self.action_num,-1)
    #     host4=host3.t()
    #     action1=action.expand(self.host_num,-1)
    #     q_value=host4+action1
        
    #     #num=self.action_num*self.host_num
    #     #q=q_value.reshape(num)
    #     q1=q_value.reshape(-1)
    #     return q1
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        h = F.relu(self.host_layer1(x))
        #host = F.relu(self.host_layer_out(h))
        host = self.host_layer_out(h)
        a = F.relu(self.action_layer1(x))
        #action = F.relu(self.action_layer_out(a))
        
        action = self.action_layer_out(a)
       
        return host,action  
    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))


    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            host,action=self.forward(x)
            a=action.max(1)[1]
            h=host.max(1)[1]
            return h,a


class DQN_PERAgent:
    """A simple Deep Q-Network Agent """

    def __init__(self,
                 env,
                 seed=None,
                 lr=0.001,
                 training_steps=10000,
                 episode_number=10000,
                 batch_size=32,
                 replay_size=5000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99,
                 hidden_sizes=[64, 64],
                 target_update_freq=1000,
                 verbose=True,
                 **kwargs):

        # This DQN implementation only works for flat actions
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning DQN with config:")
            pprint(locals())
        
        #self.ICM=ICM(self.obs_dim,self.num_actions).to(self.device)
        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # envirnment setup
        self.env = env

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape
        self.hosts=self.env.network.hosts_addresses
        self.host_num=len(self.hosts)
        self.action_num=int(self.num_actions/self.host_num)
        self.alpha_host=0.5/self.host_num
        self.alpha_action=0.5/self.action_num
        # logger setup
        self.logger = SummaryWriter()
        self.train_start=replay_size
        # Training related attributes
        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0,
                                            self.final_epsilon,
                                            self.exploration_steps)
        self.batch_size = batch_size
        self.discount = gamma
        self.training_steps = training_steps
        self.episode_number=episode_number
        self.steps_done = 0
        self.best_return=0
        self.best_action_set=[]
        self.rewards_episode=[]
        self.rewards_step=[]
        # Neural Network related attributes
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        self.dqn = DQN_dueling_noisy(self.obs_dim,
                       hidden_sizes,
                       self.action_num,self.host_num).to(self.device)
        if self.verbose:
            print(f"\nUsing Neural Network running on device={self.device}:")
            print(self.device)
            print(torch.cuda.get_device_name(0))
            print(self.dqn)
        self.num_episodes = 0
        self.target_dqn = DQN_dueling_noisy(self.obs_dim,
                              hidden_sizes,
                              self.action_num,self.host_num).to(self.device)
        self.target_update_freq = target_update_freq

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        #self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss(reduce=False)
        # PER replay
        self.replayPER = ReplayMemoryPER(replay_size,self.device)
        # replay setup
        self.replay = ReplayMemory(replay_size,
                                   self.obs_dim,
                                   self.device)
        # save sample (error,<s,a,r,s'>) to the replay memory
    # def append_sample(self, state, action, reward, next_state, done):
    #     q_vals_raw = self.dqn(torch.from_numpy(state).to(self.device))
       
    #     q_vals = q_vals_raw[action]

    #     target_q_val_raw = self.target_dqn(torch.from_numpy(next_state).to(self.device))

    #     target_q_val = target_q_val_raw.max()
    #     if done:
    #         target=reward
    #     else:
    #         target = reward + self.discount*target_q_val

    #     error = abs(q_vals  - target)

    #     self.replayPER.add(error, (state, action, reward, next_state, done))                           
    def append_sample(self, state,host, action, reward, next_state, done):
        host_q_vals_raw,action_q_vals_raw = self.dqn(torch.from_numpy(state).to(self.device))
       
        host_q_vals = host_q_vals_raw[host]
        action_q_vals=action_q_vals_raw[action]

        host_target_q_val_raw ,action_target_q_val_raw= self.target_dqn(torch.from_numpy(next_state).to(self.device))

        host_target_q_val = host_target_q_val_raw.max()
        action_target_q_val = action_target_q_val_raw.max()
        if done:
            target=reward
        else:
            target = reward + self.discount*0.5*(host_target_q_val+action_target_q_val)
        q_vals=(host_q_vals+action_q_vals)/2
        error = abs(q_vals  - target)

        self.replayPER.add(error, (state, host,action, reward, next_state, done)) 
    def save(self, save_path):
        self.dqn.save_DQN(save_path)

    def load(self, load_path):
        self.dqn.load_DQN(load_path)

    def get_epsilon(self):
        if self.num_episodes < self.exploration_steps:
            return self.epsilon_schedule[self.num_episodes]
        return self.final_epsilon



    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            o = torch.from_numpy(o).float().to(self.device)
            h,a=self.dqn.get_action(o)
            h=h.cpu().item()
            a=a.cpu().item()
            return h,a
        return random.randint(0, self.host_num-1),random.randint(0, self.action_num-1)

    def getV(self, q_value):
        v = self.alpha * torch.log((1 / self.alpha * q_value).exp().sum(dim=-1, keepdim=True))
        vv=self.alpha*torch.logsumexp(q_value/self.alpha, dim = -1) # YOUR CODE
        return vv

    def optimize(self):
        s_batch, h_batch,a_batch, next_s_batch, r_batch, d_batch, idxs, is_weight = self.replayPER.sample(self.batch_size)
        
     
        #batch = self.replay.sample_batch(self.batch_size)
        #s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        host_q_vals_raw,action_q_vals_raw = self.dqn(s_batch)
        host_q_vals = host_q_vals_raw.gather(1, h_batch).squeeze()
        action_q_vals = action_q_vals_raw.gather(1, a_batch).squeeze()
        q_vals=(action_q_vals+host_q_vals)/2
        # get target q val = max val of next state
        with torch.no_grad():
            host_target_q_val_raw,action_target_q_val_raw = self.target_dqn(next_s_batch)
            #host_target_q_val_raw,action_target_q_val_raw=self.getV(host_target_q_val_raw),self.getV(action_target_q_val_raw)
            host_target_q_val_raw2=self.alpha_host*torch.logsumexp(host_target_q_val_raw/self.alpha_host, dim = -1)
            action_target_q_val_raw2=self.alpha_action*torch.logsumexp(action_target_q_val_raw/self.alpha_action, dim = -1)
            target_q_val = (host_target_q_val_raw2 +action_target_q_val_raw2)/2
            target = r_batch + self.discount*(1-(d_batch))*target_q_val
        target=target.float()
        # calculate error
        error = torch.abs(q_vals-target_q_val).cpu().data.numpy()
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.replayPER.update(idx, error[i])
    
        # calculate loss    
        loss = (torch.cuda.FloatTensor(is_weight) * F.mse_loss(q_vals, target,reduce=False,reduction='none')).mean()
        #loss = (torch.cuda.FloatTensor(is_weight) * self.loss_fn(q_vals, target)).mean()
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.dqn.reset_noise()
        self.target_dqn.reset_noise()
        q_vals_max = action_q_vals_raw.max(1)[0]+host_q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        return loss.item(), mean_v

    def train(self):
        if self.verbose:
            print("\nStarting training")

        self.num_episodes = 1
        training_steps_remaining = self.training_steps

        while self.num_episodes < self.episode_number:
        #while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_steps_remaining)
            ep_return, ep_steps, goal = ep_results
            
            training_steps_remaining -= ep_steps
            #self.rewards_step.append(ep_return)
            if self.replayPER.tree.n_entries>=self.train_start:
                self.logger.add_scalar("episode", self.num_episodes, self.steps_done)
                
                self.logger.add_scalar(
                    "return-steps", ep_return, self.steps_done
                )
                self.logger.add_scalar(
                    "return-episode", ep_return, self.num_episodes
                )
                self.logger.add_scalar(
                    "episode_steps", ep_steps, self.steps_done
                )
                self.logger.add_scalar(
                    "episode_goal_reached", int(goal), self.steps_done
                )
                self.logger.add_scalar(
                    "episode-steps-episode", ep_steps, self.num_episodes
                )
                self.logger.add_scalar(
                    "honeypot_reached", int(self.env.honeypot_reached()), self.steps_done
                )
                self.logger.add_scalar(
                    "honeypot_reached-episodes", int(self.env.honeypot_reached()), self.num_episodes
                )
                self.num_episodes += 1
            else:
                print(f"\treplay memory: = {self.replayPER.tree.n_entries} / "
                      f"{self.train_start}")

            if self.num_episodes % 5 == 0 and self.verbose:
                print(f"\nEpisode {self.num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\tepisode steps = {ep_steps}") 
                print(f"\treturn = {ep_return}") 
                print(f"\tgoal = {goal}")

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {self.num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")
        print("最佳分数为：")
        print(self.best_return)
        for a in self.best_action_set:
            print(env.action_space.get_action(a))

    def run_train_episode(self, step_limit):
        o = self.env.reset()
        done = False

        steps = 0
        episode_return = 0
        action_set=[]
        while not done :
            eee=self.get_epsilon()
            host,action = self.get_egreedy_action(o, eee)
            a=host*self.action_num+action
            action_set.append(a)
            next_o, r, done, _ = self.env.step(a)


            #self.replay.store(o, a, next_o, r, done)   #
            #self.append_sample(o,a,r,next_o,done)
            self.append_sample(o,host,action,r,next_o,done)
            if self.replayPER.tree.n_entries>=self.train_start:
                loss, mean_v = self.optimize()


                self.steps_done += 1 
                steps += 1
                self.logger.add_scalar(
                "epsilon", eee, self.num_episodes)
                self.logger.add_scalar("loss", loss, self.steps_done)
                self.logger.add_scalar("mean_v", mean_v, self.steps_done)
            #else :
                #print( self.replayPER.tree.n_entries)
            o = next_o
            episode_return += r
            if self.replayPER.tree.n_entries>=self.train_start:
                self.rewards_episode.append(episode_return)
            if episode_return >= self.best_return  :
                    self.best_return=episode_return
                    self.best_action_set=action_set
            #steps += 1
        self.compromised_host=[]
        self.uncompromised_host=[]
        return episode_return, steps, self.env.goal_reached()

    def run_eval_episode(self,
                         env=None,
                         render=False,
                         eval_epsilon=0.05,
                         render_mode="readable"):
        if env is None:
            env = self.env
        o = env.reset()
        done = False

        steps = 0
        episode_return = 0

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render(render_mode)
            input("Initial state. Press enter to continue..")

        while not done:
            a = self.get_egreedy_action(o, eval_epsilon)
            next_o, r, done, _ = env.step(a)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                env.render(render_mode)
                print(f"Reward = {r}")
                print(f"Done = {done}")
                input("Press enter to continue..")

                if done:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")

        return episode_return, steps, env.goal_reached()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("--render_eval", action="store_true",
                        help="Renders final policy")
    parser.add_argument("-o", "--partially_obs", default=False, action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("--hidden_sizes", type=int, nargs="*",
                        default=[128,128],
                        help="(default=[64. 64])")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("-t", "--training_steps", type=int, default=1500000,
                        help="training steps (default=20000)")
    parser.add_argument("-e", "--episode_number", type=int, default=10000,
                        help="training steps (default=20000)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="(default=32)")
    parser.add_argument("--target_update_freq", type=int, default=1000,
                        help="(default=1000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="(default=0)")
    parser.add_argument("--replay_size", type=int, default=500000,
                        help="(default=100000)")
    parser.add_argument("--final_epsilon", type=float, default=0.05,
                        help="(default=0.05)")
    parser.add_argument("--init_epsilon", type=float, default=1.0,
                        help="(default=1.0)")
    parser.add_argument("--exploration_steps", type=int, default=800000,
                        help="(default=10000)")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="(default=0.99)")
    parser.add_argument("--quite", action="store_false",
                        help="Run in Quite mode")
    
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=True,
                               flat_obs=True)
    dqn_agent = DQN_PERAgent(env, verbose=args.quite, **vars(args))
    #dqn_agent.load("D:\\Experiments\\Experiment_Record\\E0812\\medium-honeypot\\saved-model\\NDSPI-D-medium-honeypot-0908.pkl")
    dqn_agent.train()
    #save_data(dqn_agent.rewards_episode,'D:\\Experiments\\NetworkAttackSimulator\\medium-multi-site\\Dqn_PER_rewards_episode_Feb25.csv')
    #save_data(dqn_agent.rewards_step,'D:\\Experiments\\NetworkAttackSimulator\\medium-multi-site\\Dqn_PER_rewards_step_Feb25.csv')
    #dqn_agent.run_eval_episode(render=args.render_eval)