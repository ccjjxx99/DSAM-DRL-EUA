import sys


def log_and_print(log_str, log_filename):
    temp = sys.stdout  # 记录当前输出指向，默认是console
    with open(log_filename, "a+") as ff:
        sys.stdout = ff  # 输出指向txt文件
        print(log_str)
        sys.stdout = temp  # 输出重定向回console
        print(log_str)
