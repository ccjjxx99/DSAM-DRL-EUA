def log_and_print(log_str, log_filename):
    print(log_str)
    with open(log_filename, "a+") as ff:
        ff.write(log_str+'\n')
