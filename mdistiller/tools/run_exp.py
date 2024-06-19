import subprocess
import itertools

# 定義你的 YAML 檔案
yaml_files = [
    'configs/cifar100/mtkd/dkd/res32x4_res8x4.yaml', 
    'configs/cifar100/mtkd/dkd/res32x4_shuv2.yaml',
    'configs/cifar100/mtkd/dkd/res50_mv2.yaml',
    'configs/cifar100/mtkd/dkd/vgg13_mv2.yaml',
    'configs/cifar100/mtkd/dkd/vgg13_vgg8.yaml',
    'configs/cifar100/mtkd/dkd/wrn40_2_wrn_16_2.yaml',
    'configs/cifar100/mtkd/dkd/wrn40_2_wrn_40_1.yaml'
    ]

# 定義你的參數
params = ['--er', '--mt', '--dt', '--ct']

# 執行指令的函數
def run_command(yaml_file, param_combination):
    command = ['pipenv', 'run', 'python3', 'tools/train.py', '--cfg', yaml_file] + list(param_combination)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Command: {' '.join(command)}")
        print(f"Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Command: {' '.join(command)}")
        print(f"Error: {e.stderr}")

# 建立所有參數組合
combinations = []
for i in range(len(params) + 1):
    combinations += itertools.combinations(params, i)

# 遍歷所有 YAML 檔案和參數組合
for yaml_file in yaml_files:
    for combination in combinations:
        run_command(yaml_file, combination)