import dataclasses

import jax
import sys
sys.path.append("/mnt/nas/wangjunbo/code/pi0/openpi/src")
sys.path.append("/mnt/nas/wangjunbo/code/lerobot")

from openpi.models import model as _model
from openpi.policies import libero_policy
from openpi.policies import Agi_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
# from openpi.training import data_loader as _data_loader

config = _config.get_config("pi0_Agi")
# checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
checkpoint_dir = "/mnt/nas/wangjunbo/code/pi0/openpi/checkpoints/60000"
# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example. 
example = Agi_policy.make_Agi_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)