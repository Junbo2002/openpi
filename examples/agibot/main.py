import dataclasses
import numpy as np
import sys
sys.path.append("/mnt/nas/wangjunbo/code/pi0/openpi/src")
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi.policies import Agi_policy


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 6666
    resize_size: int = 224
    replan_steps: int = 5


    seed: int = 7  # Random Seed (for reproducibility)

def infer(args):
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    
    element = Agi_policy.make_Agi_example()
    action_chunk = client.infer(element)["actions"]
    
    print("Actions shape:", action_chunk.shape)
    # print(action_chunk[0])


if __name__ == "__main__":
    infer(Args())