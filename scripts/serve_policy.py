import dataclasses
import enum
import logging
import socket

import tyro

import sys
# sys.path.append("/mnt/nas/wangjunbo/code/pi0/openpi/src")
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    BASE = "base"
    AGIBOT = "agibot"
    AGIBOT_FAST = "agibot_fast"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.BASE

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 6666
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
    # EnvMode.LIBERO: Checkpoint(
    #     config="pi0_libero",
    #     dir="/mnt/nas/wangjunbo/pi0/pi0_libero",
    # ),
    EnvMode.BASE: Checkpoint(
        config="pi0_aloha",
        dir="/mnt/nas/wangjunbo/pi0/pi0_base",
    ),
    EnvMode.AGIBOT: Checkpoint(
        # config="pi0_Agi_2",
        # dir="/home/wangjunbo/pi0_checkpoints/62000",
        config="pi0_agi_collected",
        # dir="/home/haichao/code/openpi/checkpoints/language/90000"  #
        # dir="/home/wangjunbo/code/openpi/checkpoints/hand/230000"  # hand
        
        # ========================
        # dir="/home/haichao/code/openpi/checkpoints/gripper/batch_2/180000"  # collected  gripper/65000 | 185000
        # dir="/home/haichao/code/openpi/checkpoints/language_from_pi0base/160000"  # LANGUAGE
        # ========================
        
        dir="/home/haichao/code/openpi/checkpoints/all/480000"  # ALL
    ),
    
    # gripper grasp everything: dir="/home/haichao/code/openpi/checkpoints/gripper/batch_2/180000"
    # pink & wooden cube 1. dir="/home/haichao/code/openpi/checkpoints/language/90000" (还是有什么抓什么)
    # pink & wooden cube 2. dir="/home/haichao/code/openpi/checkpoints/language_from_pi0base/language_from_pi0base/75000"
    
    EnvMode.AGIBOT_FAST: Checkpoint(
        config="pi0_fast_agi_collected_gripper",
        dir="/home/haichao/code/openpi/checkpoints/fast/45000" 

    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)
 

def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
