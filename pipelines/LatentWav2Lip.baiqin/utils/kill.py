import subprocess

def capture_command_output(command):
    try:
        # 使用subprocess.run执行命令，并捕获标准输出
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # 获取命令的输出字符串
        output = result.stdout
        return output
    except subprocess.CalledProcessError as e:
        # 如果命令执行出错，打印错误信息
        print(f"Error executing command: {e}")
        return None
def split_pid(output):
    # 将输出字符串按换行符分割
    lines = output.split("\n")
    # 初始化pid列表
    pids = set()
    # 遍历每一行
    for line in lines:
        # 将每一行按空格分割
        parts = line.split()
        # 如果分割后的长度大于1，说明有pid
        if len(parts) > 1:
            # 将pid添加到列表中
            pids.add(parts[1])
    return pids
def main():
    # 输入命令
    key_word='latent'
    command = f'ps -ef | grep {key_word}'
    # 捕获命令的输出
    output = capture_command_output(command)
    # 打印输出字符串
    # if output is not None:
    #     print("Command output:")
    #     print(output)
    pids = split_pid(output)
    for pid in pids:
        print(f"Killing process with PID {pid}")
        # 使用subprocess.run执行kill命令
        subprocess.run(f'kill -9 {pid}', shell=True)

if __name__ == "__main__":
    main()
