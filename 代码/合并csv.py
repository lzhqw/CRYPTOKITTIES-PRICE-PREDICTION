import pandas as pd
import os
from tqdm import tqdm


def concat_csv2():
    raw_path = '../raw data/core'
    base_name = 'sql-result-data'
    out_name = '../data/Core.csv'

    # 第一阶段 0-100

    data = pd.read_csv(os.path.join(raw_path, base_name + '.csv'))
    if len(data) == 100000:
        print(data.loc[0, 'transactions_day'])
    data.to_csv(out_name, index=False, mode='w')
    for i in tqdm(range(1, 101)):
        df = pd.read_csv(os.path.join(raw_path, base_name + ' ({})'.format(i) + '.csv'))
        if len(df) == 100000:
            print(df.loc[0, 'transactions_day'])
        df.to_csv(out_name, index=False, header=False, mode='a')
        # data = pd.concat([data, df])

    # 第二阶段 大于100
    csv_list = os.listdir(raw_path)
    csv_list = [file_name for file_name in csv_list if 'data -' in file_name]
    print(csv_list)
    for file_name in tqdm(csv_list):
        df = pd.read_csv(os.path.join(raw_path, file_name))
        if len(df) == 100000:
            print(df.loc[0, 'transactions_day'])
        df.to_csv(out_name, index=False, header=False, mode='a')
        # data = pd.concat([data, df])


def get_files(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]


def sort_files_by_ctime(folder):
    files = get_files(folder)
    files.sort(key=lambda x: os.stat(os.path.join(folder, x)).st_ctime)
    return files


def concat_csv(raw_path, out_name):
    csv_list = sort_files_by_ctime(raw_path)
    print(csv_list)
    data = pd.DataFrame()
    for csv in tqdm(csv_list):
        df = pd.read_csv(os.path.join(raw_path, csv), low_memory=False)
        if len(df) == 100000:
            print(df.loc[0, 'transactions_day'])
        data = pd.concat([data, df])
    data.to_csv(out_name, index=False)
    del data


def concat_csv_log(raw_path, out_name):
    csv_list = sort_files_by_ctime(raw_path)
    print(csv_list)
    data = pd.DataFrame()
    for csv in tqdm(csv_list):
        df = pd.read_csv(os.path.join(raw_path, csv), low_memory=False)
        if len(df) == 100000:
            print(csv)
            continue
        #     with open('dayoverlimit_log.txt', mode='a') as f:
        #         f.write(f"{df.loc[0, 'logs_day']}\n")
        data = pd.concat([data, df])
    data.drop_duplicates(inplace=True)
    data.sort_values(['logs_block_number', 'logs_log_index'], ascending=[True, True])
    data.to_csv(out_name, index=False)
    del data


if __name__ == '__main__':
    # concat_csv('../raw data/sale auction', '../data/Sale Auction.csv')
    # concat_csv('../raw data/core', '../data/Core.csv')
    concat_csv_log('../raw data/core log', '../data/log.csv')
