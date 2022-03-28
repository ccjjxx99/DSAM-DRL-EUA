from data.eua_dataset import get_dataset


def main_get_dataset():
    user_num = 600
    x_end = 0.5
    y_end = 1
    min_cov = 1
    max_cov = 1.5
    miu = 35
    sigma = 10
    data_size = {
        'train': 100000,
        'valid': 10000,
        'test': 10000
    }

    get_dataset(x_end, y_end, miu, sigma, user_num, data_size, min_cov, max_cov, device='cpu')


if __name__ == '__main__':
    main_get_dataset()
