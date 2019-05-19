import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


class QuadNextPoseIntegrator(nn.Module):

    def __init__(self):
        super(QuadNextPoseIntegrator, self).__init__()

    def forward(self, pose, velocity, delta_t):
        """
        :param pose: (num_batches, 4) (x, y, i, j)
        :param velocity: (num_batches, 3) (v, i, j)
        :param delta_t: (num_batches) (delta t)
        :return: new pose taken by time integrating velocity
        """

        v, cos, sin = velocity[:, 0], velocity[:, 1], velocity[:, 2]

        norm = torch.sqrt(torch.pow(cos, 2) + torch.pow(sin, 2))
        cos = cos / norm
        sin = sin / norm

        v_vec = torch.stack((v * cos, v * sin), dim=1)
        pose[:, :2].add(v_vec * delta_t.view(-1, 1))
        pose[:, 2] = cos
        pose[:, 3] = sin

        return pose


class Dynamics(nn.Module):
    def __init__(self, recurrent_hidden_size=32, recurrent_bias=False, linear_bias=False):
        super(Dynamics, self).__init__()
        self._hidden_size = recurrent_hidden_size

        self._velocity_cell = nn.LSTMCell(4, self._hidden_size, bias=recurrent_bias)  # takes pose (x,y,i,j)
        self._velocity_dense = nn.Linear(self._hidden_size, 3, bias=linear_bias)  # outputs velocity (v, i, j)

        self._pose_integrator = QuadNextPoseIntegrator()

        self._hidden_state = None
        self._cell_state = None

    def _no_pose_forward(self, pose):
        if self._hidden_state is None or self._cell_state is None:
            self._hidden_state, self._cell_state = self._velocity_cell(pose)
        else:
            self._hidden_state, self._cell_state = self._velocity_cell(
                pose, (self._hidden_state, self._cell_state)
            )

    def forward(self, poses, deltas, n_new_poses=1, no_reset=False):
        """
        :param poses: (seq_len, num_batches, 3) (x,y, theta)
        :param deltas: (n_new_poses, num_batches)
        :param n_new_poses: how many new poses to predict
        :param no_reset: reset hidden state
        :return: predicted poses (seq_len, num_batches, 3)
        """
        assert deltas.shape[0] == n_new_poses

        if not no_reset:
            self._hidden_state = None
            self._cell_state = None

        for pose, delta in zip(poses, deltas):
            self._no_pose_forward(pose)

        assert self._hidden_state is not None

        pred_poses = []
        last_velocity, last_pose = self._velocity_dense.forward(self._hidden_state), poses[-1]

        for num_pose, dt in zip(range(n_new_poses, deltas)):
            last_pose = self._pose_integrator.forward(pose=last_pose, delta_t=dt, velocity=last_velocity)
            pred_poses.append(last_pose)

            self._no_pose_forward(last_pose)
            last_velocity = self._velocity_dense.forward(self._hidden_state)

        return torch.stack(pred_poses)


class SeqDataset(data.Dataset):
    def __init__(self, init_samples_num: int, pred_samples_num: int, poses: np.array, t_deltas: np.array):
        """
        :param init_samples_num: number of samples to take to get hidden state
        :param pred_samples_num: number of samples to predict from hidden state
        :param poses: array of sequential poses (N, seq_len, 4) (x,y, i,j)
        :param t_deltas: array of associated time deltas (N, seq_len)
        N stand for number of such time series
        """
        assert poses.shape[0] == t_deltas.shape[0]
        self._poses = poses
        self._deltas = t_deltas

        self._init_sample_num = init_samples_num
        self._pred_sample_num = pred_samples_num

        self._len = poses.shape[0]
        self._num_possible_starts = poses.shape[1] + 1 - (self._init_sample_num + self._pred_sample_num)
        assert self._len > 0, 'no samples available if take hidden state from {} samples and predict on {}'.format(
            self._init_sample_num, self._pred_sample_num
        )

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        """
        :param item: index
        :return: tuple of (init_sample_num, 4), ((pred_sample_num, 4), (pred_sample_num#timedeltas)
        """

        start_index = np.random.randint(0, self._num_possible_starts)
        end_index = start_index + self._init_sample_num

        start_index_pred = end_index
        end_index_pred = start_index_pred + self._pred_sample_num
        return torch.Tensor(self._poses[item, start_index: end_index]), (
            torch.Tensor(self._poses[item, start_index_pred: end_index_pred]),
            torch.Tensor(self._deltas[item, start_index_pred: end_index_pred])
        )
