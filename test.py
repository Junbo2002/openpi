from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# Let's take this one for this example
repo_id = "agibotworld/task_410"
root = "/mnt/nas/Share/AgiBot/lerobot/agibotworld/task_410"

dataset = LeRobotDataset(repo_id, root=root, local_files_only=True)
# dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23])

# And see how many frames you have:
print(f"Selected episodes: {dataset.episodes}")
print(f"Number of episodes selected: {dataset.num_episodes}")
print(f"Number of frames selected: {dataset.num_frames}")
