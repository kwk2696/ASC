import argparse

from embodied_cd.environments.default.virtualhome import VirtualHomeEnv
from embodied_cd.common.print_utils import *
from scripts.dataset.generate_virtualhome import run_episode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--env_id", type=int, default=None)
    parser.add_argument("--task_id", type=int, default=None)
    return parser.parse_args()


def main(args):
    env = VirtualHomeEnv(ip=args.ip, use_unity=True)
    
    env_ids = env.valid_env_id if args.env_id is None else [args.env_id]
    task_ids = [i for i in range(30)] if args.task_id is None else [args.task_id]
    
    for env_id in env_ids:
        print(f"Env: {env_id}")
        success_tasks = []
        unsuccess_tasks = []
        for task_id in task_ids:
            print(f"Task: {task_id}")

            for room in ["kitchen", "livingroom", "bedroom", "bathroom"]:
                traj, success = [], False
                try:
                    traj, success = run_episode(env, env_id, task_id, [room], 0) 
                except Exception as e:
                    print(f"Error: {e}")
            
                if len(traj) < 2:
                    print_error("trajectory length is 1!!!")
                    success = False
                    break

            if success: success_tasks.append(task_id)
            else: unsuccess_tasks.append(task_id)
        print(f"Env {env_id}: [{success_tasks}, {unsuccess_tasks}]")
        print()


if __name__ == "__main__":
    main(parse_args())
