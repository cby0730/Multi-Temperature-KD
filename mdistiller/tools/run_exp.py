import subprocess
import itertools
import random
import multiprocessing
from multiprocessing import Queue, Process
import os

server_name = "aivc04"

# 定義你的 YAML 檔案
mtkd_yaml_files = [
    f'configs/cifar100/{server_name}/mtkd/res32x4_res8x4.yaml', 
    f'configs/cifar100/{server_name}/mtkd/res32x4_shuv2.yaml',
    f'configs/cifar100/{server_name}/mtkd/res50_mv2.yaml',
    f'configs/cifar100/{server_name}/mtkd/vgg13_mv2.yaml',
    f'configs/cifar100/{server_name}/mtkd/vgg13_vgg8.yaml',
    f'configs/cifar100/{server_name}/mtkd/wrn40_2_wrn_16_2.yaml',
    f'configs/cifar100/{server_name}/mtkd/wrn40_2_wrn_40_1.yaml'
]
mtkd_dot_yaml_files = [
    f'configs/cifar100/{server_name}/mtkd_dot/res32x4_res8x4.yaml',
    f'configs/cifar100/{server_name}/mtkd_dot/res32x4_shuv2.yaml',
    f'configs/cifar100/{server_name}/mtkd_dot/res50_mv2.yaml',
    f'configs/cifar100/{server_name}/mtkd_dot/vgg13_mv2.yaml',
    f'configs/cifar100/{server_name}/mtkd_dot/vgg13_vgg8.yaml',
    f'configs/cifar100/{server_name}/mtkd_dot/wrn40_2_wrn_16_2.yaml',
    f'configs/cifar100/{server_name}/mtkd_dot/wrn40_2_wrn_40_1.yaml'
]
dkd_yaml_files = [
    f'configs/cifar100/{server_name}/dkd/res32x4_res8x4.yaml', 
    f'configs/cifar100/{server_name}/dkd/res32x4_shuv2.yaml',
    f'configs/cifar100/{server_name}/dkd/res50_mv2.yaml',
    f'configs/cifar100/{server_name}/dkd/vgg13_mv2.yaml',
    f'configs/cifar100/{server_name}/dkd/vgg13_vgg8.yaml',
    f'configs/cifar100/{server_name}/dkd/wrn40_2_wrn_16_2.yaml',
    f'configs/cifar100/{server_name}/dkd/wrn40_2_wrn_40_1.yaml'
]
mld_yaml_files = [
    f'configs/cifar100/{server_name}/mld/res32x4_res8x4.yaml', 
    f'configs/cifar100/{server_name}/mld/res32x4_shuv2.yaml',
    f'configs/cifar100/{server_name}/mld/res50_mv2.yaml',
    f'configs/cifar100/{server_name}/mld/vgg13_mv2.yaml',
    f'configs/cifar100/{server_name}/mld/vgg13_vgg8.yaml',
    f'configs/cifar100/{server_name}/mld/wrn40_2_wrn_16_2.yaml',
    f'configs/cifar100/{server_name}/mld/wrn40_2_wrn_40_1.yaml'
]
kd_yaml_files = [
    f'configs/cifar100/{server_name}/kd/res32x4_res8x4.yaml', 
    f'configs/cifar100/{server_name}/kd/res32x4_shuv2.yaml',
    f'configs/cifar100/{server_name}/kd/res50_mv2.yaml',
    f'configs/cifar100/{server_name}/kd/vgg13_mv2.yaml',
    f'configs/cifar100/{server_name}/kd/vgg13_vgg8.yaml',
    f'configs/cifar100/{server_name}/kd/wrn40_2_wrn_16_2.yaml',
    f'configs/cifar100/{server_name}/kd/wrn40_2_wrn_40_1.yaml'
]

# 定義你的參數
params = ['--er']
mtkd_params = []

# 執行指令的函數
def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return f"Command: {' '.join(command)}\nOutput: {result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Command: {' '.join(command)}\nError: {e.stderr}"

# 建立所有參數組合
combinations = []
for i in range(len(params) + 1):
    combinations += list(itertools.combinations(params, i))

mtkd_combinations = []
for i in range(len(mtkd_params) + 1):
    mtkd_combinations += list(itertools.combinations(mtkd_params, i))

# 建立所有可能的指令
all_commands = []
for yaml_file in mtkd_yaml_files:
    for combination in mtkd_combinations:
        command = ['pipenv', 'run', 'python3', 'tools/train.py', '--cfg', yaml_file, '--er', '--mt', '--kl', '--bc'] + list(combination)
        all_commands.append(command)

'''for yaml_file in mtkd_dot_yaml_files:
    for combination in mtkd_combinations:
        command = ['pipenv', 'run', 'python3', 'tools/train.py', '--cfg', yaml_file, '--er', '--mt', '--kl', '--bc'] + list(combination)
        all_commands.append(command)'''

'''for yaml_file in dkd_yaml_files:
    for combination in combinations:
        command = ['pipenv', 'run', 'python3', 'tools/train.py', '--cfg', yaml_file] + list(combination)
        all_commands.append(command)

for yaml_file in mld_yaml_files:
    for combination in combinations:
        command = ['pipenv', 'run', 'python3', 'tools/train.py', '--cfg', yaml_file] + list(combination)
        all_commands.append(command)

for yaml_file in kd_yaml_files:
    for combination in combinations:
        command = ['pipenv', 'run', 'python3', 'tools/train.py', '--cfg', yaml_file] + list(combination)
        all_commands.append(command)'''

# 隨機打亂指令順序
#random.shuffle(all_commands)

def worker(command_queue, result_queue, worker_id):
    while True:
        command = command_queue.get()
        if command is None:
            break
        result = run_command(command)
        result_queue.put(result)

def execute_commands(commands, num_parallel):
    command_queue = Queue()
    result_queue = Queue()
    
    # 創建工作進程
    processes = []
    for i in range(num_parallel):
        p = Process(target=worker, args=(command_queue, result_queue, i))
        p.start()
        processes.append(p)
    
    # 將所有命令放入隊列
    for command in commands:
        command_queue.put(command)
    
    # 添加結束信號
    for _ in range(num_parallel):
        command_queue.put(None)
    
    # 處理結果
    completed = 0
    total = len(commands)
    while completed < total:
        result = result_queue.get()
        print(result)
        print("-" * 50)  # 分隔線
        completed += 1
        print(f"Progress: {completed}/{total}")
    
    # 等待所有進程結束
    for p in processes:
        p.join()
    
    print(f"All {total} commands have been executed.")

if __name__ == '__main__':
    num_commands_to_execute = 7  # 你可以根據需要調整這個數字
    execute_commands(all_commands, num_commands_to_execute)
