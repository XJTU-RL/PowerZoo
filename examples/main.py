import argparse

def main():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description='PowerZoo Main Script')

    # 添加参数
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--log', type=str, help='Path to the log file')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'eval'], help='Mode to run the script in')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to run')
    
    # 解析参数
    args = parser.parse_args()

    # 打印解析到的参数
    print(args)

if __name__ == "__main__":
    main()
