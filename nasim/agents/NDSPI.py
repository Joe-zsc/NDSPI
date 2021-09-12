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
import math
import numpy as np
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
        a=torch.from_numpy(np.vstack(list(batchs[1])).astype(np.int64)).to(self.device)    
        r=torch.from_numpy(np.array(list(batchs[2]))).to(self.device)
        s_=torch.from_numpy(np.vstack(batchs[3])).to(self.device)
        d=torch.from_numpy(np.array(list(batchs[4])).astype(np.int32)).to(self.device)
        return s,a,s_,r,d, idxs, is_weight

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
class ICM(nn.Module):
    def __init__(self, input_dim, num_actions):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(ICM, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.fc4 = nn.Linear(7 * 7 * 64, 512)
        dd=128
        self.linear1=nn.Linear(input_dim[0],dd)
        self.linear2=nn.Linear(dd,dd)
        self.linear3=nn.Linear(dd,dd)
        
        self.pred_module1 = nn.Linear(dd + num_actions, dd)
        self.pred_module2 = nn.Linear(dd, dd)
            
        self.invpred_module1 = nn.Linear(dd + dd, dd)
        self.invpred_module2 = nn.Linear(dd, num_actions)

    def get_feature(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        #x = F.relu(self.linear3(x))
        return x
    
    def forward(self, x):
        # get feature
        feature_x = self.get_feature(x)
        return feature_x
    
    def get_full(self, x, x_next, a_vec):
        # get feature
        feature_x = self.get_feature(x)
        feature_x_next = self.get_feature(x_next)

        pred_s_next = self.pred(feature_x, a_vec) # predict next state feature
        pred_a_vec = self.invpred(feature_x, feature_x_next) # (inverse) predict action
 
        return pred_s_next, pred_a_vec, feature_x_next

    def pred(self, feature_x, a_vec):
        # Forward prediction: predict next state feature, given current state feature and action (one-hot)
        aaa=torch.cat([feature_x, a_vec.float()], dim = -1)
        bbb=aaa.detach()
        pred_s_next = F.relu(self.pred_module1( bbb))
        pred_s_next = self.pred_module2(pred_s_next)
        return pred_s_next
    
    def invpred(self,feature_x, feature_x_next):
        # Inverse prediction: predict action (one-hot), given current and next state features
        pred_a_vec = F.relu(self.invpred_module1(torch.cat([feature_x, feature_x_next], dim = -1)))
        pred_a_vec = self.invpred_module2(pred_a_vec)
        return F.softmax(pred_a_vec, dim = -1)

class NoisyDoubleDuelingDQN(nn.Module):
    """A simple Deep Q-Network """

    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        '''
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(NoisyLinear(layers[l-1], layers[l]))
        #self.out = NoisyLinear(layers[-1], num_actions)
        self.value_stream=NoisyLinear(layers[-1],1)
        self.advantage_stream=NoisyLinear(layers[-1], num_actions)
        '''
        n=256
        self.linear =  nn.Linear(input_dim[0], n)
        self.linear2 =  NoisyLinear(n, n)
        self.advantage_hidden_layer = NoisyLinear(n, n)
        self.advantage_layer = NoisyLinear(n, num_actions)

        self.value_hidden_layer = NoisyLinear(n, n)
        self.value_layer = NoisyLinear(n, 1)
    def forward(self, x):
        
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        value=F.relu(self.value_hidden_layer(x))
        values = self.value_layer(value)

        advantages = F.relu(self.advantage_hidden_layer(x))
        advantages = self.advantage_layer(advantages)

        qvals = values + (advantages - advantages.mean())
        return qvals

    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def get_action(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            return self.forward(x).max(1)[1]
    def reset_noise(self):
        self.linear2.reset_noise()
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

class NoisyDoubleDuelingDQN_PERAgent:
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
                 forward_scale = 0.8,
                 inverse_scale = 0.2,
                 Qloss_scale = 1,
                 intrinsic_scale= 1,
                 use_extrinsic = True,
                 use_episode = True,
                 **kwargs):
        self.use_episode=use_episode
        # This DQN implementation only works for flat actions
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning DQN with config:")
            pprint(locals())

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # envirnment setup
        self.env = env
     ##
        self.compromised_host=[]
        self.uncompromised_host=[]
        self.hosts=self.env.network.hosts_addresses
        
        ##
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape
        self.alpha=0.5/self.num_actions
        # logger setup
        self.logger = SummaryWriter()

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
        self.train_start=replay_size
        self.best_return=0
        self.best_action_set=[]
        self.rewards_episode=[]
        self.rewards_step=[]
        self.num_episodes = 0
        # Neural Network related attributes
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")
        self.dqn = NoisyDoubleDuelingDQN(self.obs_dim,
                       hidden_sizes,
                       self.num_actions).to(self.device)

        # param for ICM
        self.forward_scale = forward_scale # scale for loss function of forward prediction model, 0.8        
        self.inverse_scale = inverse_scale # scale for loss function of inverse prediction model, 0.2
        self.Qloss_scale = Qloss_scale # scale for loss function of Q value, 1
        self.intrinsic_scale = intrinsic_scale # scale for intrinsic reward, 1
        self.use_extrinsic = use_extrinsic # whether use extrinsic rewards, if False, only intrinsic reward generated from ICM is used
        self.ICM=ICM(self.obs_dim,self.num_actions).to(self.device)

        self.target_dqn = NoisyDoubleDuelingDQN(self.obs_dim,
                              hidden_sizes,
                              self.num_actions).to(self.device)
        self.target_update_freq = target_update_freq
        if self.verbose:
            print(f"\nUsing Neural Network running on device={self.device}:")
            print(self.device)
            print(torch.cuda.get_device_name(0))
            print(self.dqn)
            print(self.ICM)
        self.optimizer = optim.Adam(list(self.dqn.parameters())+list(self.ICM.parameters()), lr=self.lr)
        #self.loss_fn = nn.SmoothL1Loss()
        self.loss_fn = nn.MSELoss(reduce=False)
        # PER replay
        self.replayPER = ReplayMemoryPER(replay_size,self.device)
        # replay setup
          
          
        # save sample (error,<s,a,r,s'>) to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        q_vals_raw = self.dqn(torch.from_numpy(state).to(self.device))
       
        q_vals = q_vals_raw[action]

        target_q_val_raw = self.target_dqn(torch.from_numpy(next_state).to(self.device))

        target_q_val = target_q_val_raw.max()
        if done:
            target=reward
        else:
            target = reward + self.discount*target_q_val

        error = abs(q_vals  - target)

        self.replayPER.add(error, (state, action, reward, next_state, done))                           

    def save(self, save_path):
        self.dqn.save_DQN(save_path)

    def load(self, load_path):
        self.dqn.load_DQN(load_path)

    def get_epsilon(self):
        if self.use_episode:
            if self.num_episodes < self.exploration_steps:
                return self.epsilon_schedule[self.num_episodes]
            return self.final_epsilon
        else:
            if self.steps_done < self.exploration_steps:
                return self.epsilon_schedule[self.steps_done]
            return self.final_epsilon
    def get_epsilon2(self):
        if self.steps_done < self.exploration_steps:
            a=self.exploration_steps
            f=self.final_epsilon
            w=math.pow(a,2)
            x=self.steps_done
            y=(w-(1-f)*x*x)/w
            return y
        return self.final_epsilon
    
    def get_action_set(self):
        #去除
        state=self.env.current_state.copy()
        actionset=[]
        test=[]
        s=[]
        self.uncompromised_host=[i for i in self.hosts if i not in self.compromised_host] 
        #除去已经获取权限的目标
        for addr in self.uncompromised_host:
            if  state.host_has_access(addr, 2):
                self.compromised_host.append(addr)
        #for addr in self.uncompromised_host:

        
        for a in range(self.env.action_space.n):
            t=self.env.action_space.get_action(a)
            #对未获取权限的目标
            #if t.target not in s:
                
            if t.target in self.uncompromised_host:
                if t.target not in s:
                    s.append(t.target)
                    b=state.get_host(t.target)
                    test.append(b)
                    #如果可达
                if state.host_reachable(t.target) or state.host_discovered(t.target):
                        #if (b.access ==0 and t.name.find('scan')!= -1) \
                            #or (t.name.find('scan')== -1 and b.access>=t.req_access):
                            #if t.name.find('pe') != -1 :
                                #a=1
                        actionset.append(a)
        #for a in actionset:
            #print(env.action_space.get_action(a))
            #print(a)
        if len(actionset)==0:
            self.env.render_network_graph(show=True)
            x=1
        test=[]
        s=[]
        return actionset

    def get_egreedy_action(self, o, epsilon):
        if random.random() > epsilon:
            o = torch.from_numpy(o).float().to(self.device)
            return self.dqn.get_action(o).cpu().item()
        return random.randint(0, self.num_actions-1)

    def get_egreedy_action2(self, o, epsilon):
        
        if random.random() > epsilon:
            o = torch.from_numpy(o).float().to(self.device)
            return self.dqn.get_action(o).cpu().item()
        actions=self.get_action_set()
        x=random.randint(0, len(actions)-1)
        return actions[x]
    def getV(self, q_value):
        #v = self.alpha * torch.log((1 / self.alpha * q_value).exp().sum(dim=-1, keepdim=True))
        vv=self.alpha*torch.logsumexp(q_value/self.alpha, dim = -1) # YOUR CODE
        return vv
    def optimize(self):
        s_batch, a_batch, next_s_batch, r_batch, d_batch, idxs, is_weight = self.replayPER.sample(self.batch_size)
        
     
        #batch = self.replay.sample_batch(self.batch_size)
        #s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.dqn(s_batch)
        
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()

        #ICM
        a_vec=F.one_hot(a_batch.squeeze(), num_classes = self.num_actions).to(self.device)# convert action from int to one-hot format
        pred_s_next, pred_a_vec, feature_x_next = self.ICM.get_full(s_batch, next_s_batch, a_vec)
        forward_loss = F.mse_loss(pred_s_next, feature_x_next.detach(), reduce=False,reduction='none')
        inverse_pred_loss = F.cross_entropy(pred_a_vec, a_batch.squeeze().detach(), reduce=False,reduction='none')
        # calculate rewards
        intrinsic_rewards = self.intrinsic_scale * forward_loss.mean(-1)
        total_rewards = intrinsic_rewards.clone()
        if self.use_extrinsic:
            total_rewards += r_batch
        r_batch=total_rewards
        

        # get target q val = max val of next state
        with torch.no_grad():
            # argmax_Q = self.dqn(next_s_batch).max(1)[1].unsqueeze(1)       
            # DDQNtarget_q_val = self.target_dqn(next_s_batch).gather(1,argmax_Q).squeeze(1)
            #target_q_val_raw = self.target_dqn(next_s_batch)
            #target_q_val = target_q_val_raw.max(1)[0]
            target_q_val_raw = self.target_dqn(next_s_batch)
            #target_q_val = target_q_val_raw.max(1)[0]
            target_q_val = self.getV(target_q_val_raw)
            target = r_batch + self.discount*(1-(d_batch))*target_q_val
        target=target.float()
        # calculate error
        error = torch.abs(q_vals-target_q_val).cpu().data.numpy()
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.replayPER.update(idx, error[i])
    
        # calculate loss    
        Q_loss = (torch.cuda.FloatTensor(is_weight) * F.mse_loss(q_vals, target,reduce=False,reduction='none')).mean()
        loss = self.Qloss_scale*Q_loss + self.forward_scale*forward_loss.mean() + self.inverse_scale* inverse_pred_loss.mean()
        #loss = (torch.cuda.FloatTensor(is_weight) * self.loss_fn(q_vals, target)).mean()
        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        self.dqn.reset_noise()
        self.target_dqn.reset_noise()
        q_vals_max = q_vals_raw.max(1)[0]
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
            #ep_return, ep_steps, goal, honeypot_state = ep_results
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
                # self.logger.add_scalar(
                #     "honeypot_reached", int(self.env.honeypot_reached()), self.steps_done
                # )
                # self.logger.add_scalar(
                #     "honeypot_reached-episodes", int(self.env.honeypot_reached()), self.num_episodes
                # )
                self.num_episodes += 1
            else:
                print(f"\treplay memory: = {self.replayPER.tree.n_entries} / "
                      f"{self.train_start}")

            if self.num_episodes % 5 == 0 and self.verbose:
                print(f"\nEpisode {self.num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\tepisode steps = {ep_steps}") 
                print(f"\tepsilon = {self.get_epsilon()}") 
                print(f"\treturn = {ep_return}")
                print(f"\tbest return = {self.best_return}") 
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
        while not done and steps < step_limit:
            eee=self.get_epsilon()
            a = self.get_egreedy_action(o, eee)
            action_set.append(a)
            next_o, r, done, _ = self.env.step(a)


            #self.replay.store(o, a, next_o, r, done)   #
            self.append_sample(o,a,r,next_o,done)

            if self.replayPER.tree.n_entries>=self.train_start:
                loss, mean_v = self.optimize()


                self.steps_done += 1 
                steps += 1
                self.logger.add_scalar(
                "epsilon", eee, self.num_episodes)
            
                self.logger.add_scalar("loss", loss, self.num_episodes)
                self.logger.add_scalar("mean_v", mean_v, self.num_episodes)
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
        #return episode_return, steps, self.env.goal_reached(), self.env.honeypot_reached()
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
    parser.add_argument("--batch_size", type=int, default=128,
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
    #env.render_network_graph(show=True)
    dqn_agent =  NoisyDoubleDuelingDQN_PERAgent(env, verbose=args.quite, **vars(args))
    num_actions = env.action_space.n
    #for a in range(num_actions):
          #  print(env.action_space.get_action(a))
    dqn_agent.train()
    dqn_agent.save("D:\\Experiments\\Experiment_Record\\E0812\\Saved_model\\NDSPI-medium-multi-site-0820.pkl")
    #dqn_agent.save("D:\\Experiments\\NetworkAttackSimulator\\medium-multi-site-honeypot\\NoisyDoubleDueling_PER_Mar15.pkl")
    #save_data(dqn_agent.rewards_episode,'D:\\Experiments\\NetworkAttackSimulator\\medium-multi-site\\ NoisyDoubleDueling_PER_rewards_episode_Mar1.csv')
    #save_data(dqn_agent.rewards_step,'D:\\Experiments\\NetworkAttackSimulator\\medium-multi-site\\ NoisyDoubleDueling_PER_rewards_step_Mar1.csv')
   #dqn_agent.run_eval_episode(render=args.render_eval)