import torch
from torch import nn
from functools import reduce
import numpy as np

from q1_schedule import LinearExploration, LinearSchedule
from utils.test_env import EnvTest
from core.q_learning import QN

from configs.q2_linear import config

class DQN_PT(QN):
    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = torch.tensor(state, dtype=torch.float32)
        state /= self.config.high


        if len(state.shape) < 4:
            state = state.flatten()
            state.unsqueeze_(0)
        else:
            state = state.flatten(1)

        return state

    def build(self):
        """
        Build model
        """
        pass

    def initialize(self):
        """
        Initialize the stuff
        """
        state_shape = list(self.env.observation_space.shape)
        state_shape[-1] = self.config.state_history
        
        # build model
        in_dim = reduce(lambda a,b: a*b, state_shape)

        num_actions = self.env.action_space.n

        self.model = nn.Linear(in_dim, num_actions).cuda()
        self.target_model = nn.Linear(in_dim, num_actions).cuda()
        self.update_target_params()

        # optimization
        self.loss = lambda a, b: torch.mean((a - b) ** 2)
        self.optim = torch.optim.Adam(self.model.parameters(), 0.0001)
        self.optim.zero_grad()

    def save(self):
        """
        Save session
        """
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
        }, self.config.model_output)

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        state = self.process_state(state).cuda()

        action_values = self.model(state).flatten().cpu().detach_().numpy()

        best_actions = np.argmax(action_values)

        return best_actions, action_values

    def _get_grad(self):
        w = self.model.weight.grad
        b = self.model.bias.grad

        if w is None or b is None:
            return 0

        grad = torch.cat([w.flatten(), b.flatten()])

        return grad.detach()

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Q_samp(s) = r if done
                    = r + gamma * max_a' Q_target(s', a')
        loss = (Q_samp(s) - Q(s, a))^2 

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)
        
        s_batch = self.process_state(s_batch).cuda()
        sp_batch = self.process_state(sp_batch).cuda()

        # a_batch = torch.Tensor(r_batch).cuda().long()
        r_batch = torch.tensor(r_batch).cuda()
        done_mask_batch = torch.tensor(done_mask_batch).cuda()

        # forward
        pred_qval = self.model(s_batch)
        pred_qval = torch.stack([pred_qval[i][a_batch[i]] for i in range(len(pred_qval))])
        next_qval = self.target_model(sp_batch).max(1)[0] * (1 - done_mask_batch)

        # optimize
        target_qval = r_batch + self.config.gamma * next_qval

        loss = self.loss(target_qval, pred_qval)

        self.optim.param_groups[0]['lr'] = lr

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss, torch.norm(self._get_grad())

    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        state_dic = self.model.state_dict()
        self.target_model.load_state_dict(state_dic)

if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = DQN_PT(env, config)
    model.run(exp_schedule, lr_schedule)