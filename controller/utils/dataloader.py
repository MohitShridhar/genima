import torch
from torch.utils.data import Dataset
import numpy as np
from robobase.replay_buffer.uniform_replay_buffer import UniformReplayBuffer, episode_len
from robobase.replay_buffer.uniform_replay_buffer import (
    REWARD,
    ACTION,
    TERMINAL,
    TRUNCATED,
    INDICES,
    DISCOUNT,
)
import traceback


class EpochReplayBuffer(UniformReplayBuffer, Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_start = 0

    @property
    def length(self):
        return np.minimum(self._add_count.value, self._replay_capacity)

    def _sample(self, global_index=None):
        # index here is the "global" index of a flattened sample
        self._samples_since_last_fetch += 1
        if global_index not in self._global_idxs_to_episode_and_transition_idx:
            # This worker does not have this sample
            print(f"in _sample, not in: {global_index}")
            return None
        (
            episode_fn,
            transition_idx,
        ) = self._global_idxs_to_episode_and_transition_idx[global_index]
        episode = self._episodes[episode_fn]
        idx = transition_idx
        ep_len = episode_len(episode)
        next_idx = idx + self._nstep
        if next_idx > episode[REWARD].shape[0]:
            next_idx = episode[REWARD].shape[0]
        start_idx = (idx - self._frame_stack) + 1
        start_next_idx = (next_idx - self._frame_stack) + 1
        # Turn all negative idxs to 0
        obs_idxs = list(map(lambda x: np.clip(x, 0, ep_len), range(start_idx, idx + 1)))
        obs_next_idxs = list(
            map(lambda x: np.clip(x, 0, ep_len), range(start_next_idx, next_idx + 1))
        )

        discount_slice_len = next_idx - idx
        replay_sample = {
            REWARD: np.sum(
                episode[REWARD][idx:next_idx]
                * self._cumulative_discount_vector[
                    : episode[REWARD][idx:next_idx].shape[0]
                ]
            ),
            ACTION: episode[ACTION][idx],
            TERMINAL: episode[TERMINAL][next_idx - 1],
            TRUNCATED: episode[TRUNCATED][next_idx - 1],
            INDICES: global_index,
            DISCOUNT: self._gamma**discount_slice_len,  # effective discount
        }
        # Add remaining (extra) items
        for name in self._storage_signature.keys():
            if name not in replay_sample:
                replay_sample[name] = episode[name][idx]
        for name in self._obs_signature.keys():
            replay_sample[name] = episode[name][obs_idxs]
            if not self._sequential:
                # Sequential buffer does not need tp1 observations
                replay_sample[name + "_tp1"] = episode[name][obs_next_idxs]
        return replay_sample

    def __next__(self):
        try:
            self._try_fetch()
        except Exception as e:
            print(e)
            traceback.print_exc()
        indices = torch.randperm(self.length)
        indices = [
            i.item()
            for i in indices
            if i.item() in self._global_idxs_to_episode_and_transition_idx
        ]

        if (self.batch_start + self.batch_size - 1) >= self.length:
            raise StopIteration

        batch_indices = indices[self.batch_start : self.batch_start + self.batch_size]
        self.batch_start += self.batch_size
        return self.sample(batch_size=len(batch_indices), indices=batch_indices)

    def __iter__(self):
        self.batch_start = 0
        return self
