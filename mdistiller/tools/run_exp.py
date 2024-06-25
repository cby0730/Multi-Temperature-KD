import subprocess
import itertools
import random
import concurrent.futures
import os

# 定義你的 YAML 檔案
yaml_files = [
    'configs/cifar100/mtkd_exp/res32x4_res8x4.yaml', 
    'configs/cifar100/mtkd_exp/res32x4_shuv2.yaml',
    'configs/cifar100/mtkd_exp/res50_mv2.yaml',
    'configs/cifar100/mtkd_exp/vgg13_mv2.yaml',
    'configs/cifar100/mtkd_exp/vgg13_vgg8.yaml',
    'configs/cifar100/mtkd_exp/wrn40_2_wrn_16_2.yaml',
    'configs/cifar100/mtkd_exp/wrn40_2_wrn_40_1.yaml'
]

# 定義你的參數
params = ['--er', '--mt', '--dt', '--bc', '--ct', '--std']

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

# 建立所有可能的指令
all_commands = []
for yaml_file in yaml_files:
    for combination in combinations:
        command = ['pipenv', 'run', 'python3', 'tools/train.py', '--cfg', yaml_file] + list(combination)
        all_commands.append(command)

# 隨機打亂指令順序
random.shuffle(all_commands)

def execute_commands(commands, num_commands):
    # 如果指定的數量大於可用的指令數量，就使用所有可用的指令
    num_commands = min(num_commands, len(commands))
    
    print(f"Executing {num_commands} commands out of {len(commands)} total commands.")
    
    # 使用進程池來並行執行指令
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_command, command) for command in commands[:num_commands]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            print("-" * 50)  # 分隔線
    
    print(f"Executed {num_commands} commands.")

if __name__ == '__main__':
    num_commands_to_execute = 3  # 你可以根據需要調整這個數字
    execute_commands(all_commands, num_commands_to_execute)