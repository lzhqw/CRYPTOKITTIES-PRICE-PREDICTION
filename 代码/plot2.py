import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

sns.set_theme(style="darkgrid")

'''
PART 1 频次图
    1. 挑选出input data 中以函数地址开头的行
    2. 将时间转换为时间格式
    3. 调用seabron histplot绘制频次图，stat默认为count，故未进行设置
    4. 使用对数坐标以使得高度差异不是太大
'''
def plot_cSaA(data):
    '''
    绘制createSaleAuction函数的调用频次
    '''
    data = data.loc[data['transactions_input'].str.startswith('0x3d7d3f5a')]
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    fig = sns.histplot(data=data, x='transactions_day', binwidth=14)
    fig.set_xlim(pd.to_datetime('2017-11-01'), pd.to_datetime('2023-03-01'))
    plt.yscale("log")
    if ifShow:
        plt.show()
    else:
        plt.savefig('saleauction count.pdf', dpi=300)
    print('\033[1;32m -- saleauction count done! -- \033[0m')


def plot_cSiA(data):
    '''
    绘制createSiringAuction函数的调用频次
    '''
    data = data.loc[data['transactions_input'].str.startswith('0x4ad8c938')]
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    fig = sns.histplot(data=data, x='transactions_day', binwidth=14)
    fig.set_xlim(pd.to_datetime('2017-11-01'), pd.to_datetime('2023-03-01'))
    plt.yscale("log")
    if ifShow:
        plt.show()
    else:
        plt.savefig('siringauction count.pdf', dpi=300)
    print('\033[1;32m -- siringauction count done! --\033[0m')


def plot_gB(data):
    '''
    绘制giveBirth函数的调用频次
    '''
    data = data.loc[data['transactions_input'].str.startswith('0x88c2a0bf')]
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    fig = sns.histplot(data=data, x='transactions_day', binwidth=14)
    fig.set_xlim(pd.to_datetime('2017-11-01'), pd.to_datetime('2023-03-01'))
    plt.yscale("log")
    if ifShow:
        plt.show()
    else:
        plt.savefig('givebrith count.pdf', dpi=300)
    print('\033[1;32m -- givebrith count done! --\033[0m')


def plot_bWA(data):
    '''
    绘制breedWithAuto函数的调用频次
    '''
    data = data.loc[data['transactions_input'].str.startswith('0xf7d8c883')]
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    fig = sns.histplot(data=data, x='transactions_day', binwidth=14)
    fig.set_xlim(pd.to_datetime('2017-11-01'), pd.to_datetime('2023-03-01'))
    plt.yscale("log")
    if ifShow:
        plt.show()
    else:
        plt.savefig('breedWithAuto count.pdf', dpi=300)
    print('\033[1;32m -- breedWithAuto count done! --\033[0m')


def plot_bid_sir(data):
    '''
    绘制bidOnSiring函数的调用频次
    '''
    data = data.loc[data['transactions_input'].str.startswith('0xed60ade6')]
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    fig = sns.histplot(data=data, x='transactions_day', binwidth=14)
    fig.set_xlim(pd.to_datetime('2017-11-01'), pd.to_datetime('2023-03-01'))
    plt.yscale("log")
    if ifShow:
        plt.show()
    else:
        plt.savefig('bidOnSA count.pdf', dpi=300)
    print('\033[1;32m -- bidOnSA count done! --\033[0m')


def plot_bid(data):
    '''
    绘制bidOnSiring函数的调用频次
    '''
    data = data.loc[data['transactions_input'].str.startswith('0x454a2ab3')]
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    fig = sns.histplot(data=data, x='transactions_day', binwidth=14)
    fig.set_xlim(pd.to_datetime('2017-11-01'), pd.to_datetime('2023-03-01'))
    plt.yscale("log")
    if ifShow:
        plt.show()
    else:
        plt.savefig('bid count.pdf', dpi=300)
    print('\033[1;32m -- bid count done! --\033[0m')


def plot_createMinusBid():
    '''
    绘制createSaleAuction频次与bid频次的差
    1. 通过value_counts 获得频次的统计信息
    2. 以日期作为索引
    3. 合并两个表以对齐日期
    4. 做差
    5. 调用scatterplot绘制散点图
    6. 将y轴控制在-500 ~ 500之间，以看清接近0的部分
    '''
    saleAuction = pd.read_csv('../data/sale auction.csv')
    bid = saleAuction.loc[saleAuction['transactions_input'].str.startswith('0x454a2ab3')]
    bid_count = pd.DataFrame(bid['transactions_day'].value_counts())
    bid_count.columns = ['bid']
    del saleAuction, bid
    core = pd.read_csv('../data/Core.csv')
    createSale = core.loc[core['transactions_input'].str.startswith('0x3d7d3f5a')]
    createSale_count = pd.DataFrame(createSale['transactions_day'].value_counts())
    createSale_count.columns = ['createSale']
    del core, createSale

    data = pd.concat([bid_count, createSale_count], axis=1)

    data['transactions_day'] = data.index
    print(data)
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    data['diff'] = data['bid'] - data['createSale']
    sns.scatterplot(data=data, x='transactions_day', y='diff', size=5)
    plt.ylim(-500, 500)
    if ifShow:
        plt.show()
    else:
        plt.savefig('createMinusBid.pdf', dpi=1200)
    print('\033[1;32m -- ceateMinusBid done! --\033[0m')


def _plot_price(data, col):
    '''
    绘制价格的折线图（绘图部分）
    1. 调用seaborn的lineplot进行绘图
    2. y轴以对数坐标显示
    :param data: transaction 表
    :param col: 列名
    :return:
    '''
    plt.figure(figsize=(16, 6))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    fig = sns.lineplot(data=data, x='transactions_day', y=col, size=1)
    fig.set_xlim(pd.to_datetime('2017-11-01'), pd.to_datetime('2023-03-01'))
    plt.yscale("log")
    if ifShow:
        plt.show()
    else:
        plt.savefig('price {}.pdf'.format(col), dpi=300)
    print('\033[1;32m -- price {} done! --\033[0m'.format(col))


def plot_price(data, price):
    '''
    绘制价格的折线图（数据处理部分）
    :param data: transactions表 合约为Core
    :param price: dict，以太币-美元的汇率
    :return:
    '''
    data = data.loc[data['transactions_input'].str.startswith('0x3d7d3f5a')]
    # 有5个数据长度不是266，但经查证状态为Fail
    # data = data[data['transactions_input'].str.len() != 266]
    # data.reset_index(inplace=True)
    # for i in range(len(data)):
    #     print(data.loc[i, 'transactions_input'])
    #     print(data.loc[i, 'transactions_hash'])
    # -------------------------------------------------------- #
    # step 1. 校验input data长度是否正确
    # -------------------------------------------------------- #
    data = data[data['transactions_input'].str.len() == 266]
    data.reset_index(inplace=True)
    # -------------------------------------------------------- #
    # step 2. 从input data中解析输入参数（每64位表示一个16进制数）
    # -------------------------------------------------------- #
    data['kittyId'] = data['transactions_input'].map(lambda x: int(x[10:10 + 64], base=16))
    data['ETHsP'] = data['transactions_input'].map(lambda x: int(x[10 + 64:10 + 64 * 2], base=16))
    data['ETHeP'] = data['transactions_input'].map(lambda x: int(x[10 + 64 * 2:10 + 64 * 3], base=16))
    data['duration'] = data['transactions_input'].map(lambda x: int(x[10 + 64 * 3:10 + 64 * 4], base=16))
    # -------------------------------------------------------- #
    # step 3. 换算为美元
    # -------------------------------------------------------- #
    data['price'] = data['transactions_day'].map(lambda x: price[x])
    data['USDsP'] = data['ETHsP'] * data['price'] * 10 ** -18  # input data中的单位是Wei, 1ETH = 10^18 Wei
    data['USDeP'] = data['ETHeP'] * data['price'] * 10 ** -18
    # -------------------------------------------------------- #
    # step 4. 删掉一些太大的/胡乱出价
    # -------------------------------------------------------- #
    data = data.loc[data['USDsP'] < 1e10]
    data = data.loc[data['USDeP'] < 1e10]
    # -------------------------------------------------------- #
    # step 4.5 绘制不同拍卖的比例
    # -------------------------------------------------------- #
    cnt1, cnt2, cnt3 = 0, 0, 0
    for i in tqdm(data.index):
        if data.loc[i, 'USDsP'] > data.loc[i, 'USDeP']:
            cnt1 += 1
        elif data.loc[i, 'USDsP'] == data.loc[i, 'USDeP']:
            cnt2 += 1
        else:
            cnt3 += 1
    cnt1 = cnt1 / len(data)
    cnt2 = cnt2 / len(data)
    cnt3 = cnt3 / len(data)
    print(cnt1, cnt2, cnt3)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0, bottom=0, right=0.85, top=1)
    plt.pie(x=[cnt1, cnt2, cnt3], labels=['荷兰式拍卖', '定价拍卖', '增价拍卖'],
            colors=['#384871', '#E7BD39', '#AB4F3F'],
            explode=[0.02 for i in range(3)], startangle=90)
    plt.legend(bbox_to_anchor=(0.95, 0.6))
    if ifShow:
        plt.show()
    else:
        plt.savefig('bid pie.svg', dpi=300)

    # 按月为单位的话用下面这行代码 2023-02-08 ——> 2023-02
    # data['transactions_day'] = data['transactions_day'].map(lambda x: x[:7])
    # -------------------------------------------------------- #
    # step 5. 时间格式转换
    # -------------------------------------------------------- #
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    # -------------------------------------------------------- #
    # step 6. 画图
    # -------------------------------------------------------- #
    _plot_price(data, 'USDsP')
    _plot_price(data, 'USDeP')
    _plot_price(data, 'ETHsP')
    _plot_price(data, 'ETHeP')


def plot_bid_price(data, price):
    '''
    绘制拍卖成功的交易价格折线图（数据处理部分）
    :param data: transactions 表 SaleAuction
    :param price: dict，以太币-美元的汇率
    :return:
    '''
    print(len(data))
    # -------------------------------------------------------- #
    # step 1. 挑选出bid函数对应的行
    # -------------------------------------------------------- #
    data = data.loc[data['transactions_input'].str.startswith('0x454a2ab3')]
    data = data[data['transactions_input'].str.len() == 74]
    print(len(data))
    # -------------------------------------------------------- #
    # step 2. 解析input data
    # -------------------------------------------------------- #
    data['kittyId'] = data['transactions_input'].map(lambda x: int(x[10:], base=16))
    data['price'] = data['transactions_day'].map(lambda x: price[x])
    data['transactions_value'] = data['transactions_value'].map(lambda x: int(x))
    data['USDP'] = data['transactions_value'] * data['price'] * 10 ** -18
    data.loc[:, 'transactions_day'] = pd.to_datetime(data['transactions_day'])
    # -------------------------------------------------------- #
    # step 3. 绘图
    # -------------------------------------------------------- #
    _plot_price(data, 'USDP')
    _plot_price(data, 'transactions_value')


def plot_eth():
    '''
    绘制以太币的增幅图
    :return:
    '''
    price_data = pd.read_csv('../data/eth price.csv')
    print(price_data)

    price_data.drop('id', axis=1, inplace=True)
    price_data.sort_values('create_date', ascending=True, inplace=True)
    print(price_data)

    price_data['日增幅'] = None
    price_data['月增幅'] = None
    for i in range(365 * 2, len(price_data)):
        if i >= 1:
            price_data.loc[i, '日增幅'] = (price_data.loc[i, 'eth_price'] - price_data.loc[i - 1, 'eth_price']) / \
                                       price_data.loc[i - 1, 'eth_price']
        if i >= 30:
            price_data.loc[i, '月增幅'] = (price_data.loc[i, 'eth_price'] - price_data.loc[i - 30, 'eth_price']) / \
                                       price_data.loc[i - 30, 'eth_price']
    price_data.loc[:, 'create_date'] = pd.to_datetime(price_data['create_date'])
    price_data = price_data.loc[365 * 2 + 30:len(price_data), :]
    plt.bar(x=price_data['create_date'], height=price_data['月增幅'])
    plt.show()


def _pie_plot_user(function_list, name):
    '''
    绘制不同调用函数频次的用户群体的调用函数的比例（绘图部分）
    :param function_list: dict，每个函数调用了多少次
    :param name: 输出名称
    :return:
    '''
    print(function_list)
    # -------------------------------------------------------- #
    # step 0. 为保证一致性，指定每个函数对应的颜色
    # -------------------------------------------------------- #
    colors = {'0x095ea7b3': color_list[0], '0x454a2ab3': color_list[1], '0xa9059cbb': color_list[2],
              '0xf7d8c883': color_list[3], '0xed60ade6': color_list[4], '0x3d7d3f5a': color_list[5],
              '0x4ad8c938': color_list[6], '0x88c2a0bf': color_list[7], 'others': color_list[8]}
    # -------------------------------------------------------- #
    # step 1. 将调用比例小于5%的函数归为others
    # -------------------------------------------------------- #
    function_count = sum(function_list.values())
    function_list.setdefault('others', 0)
    drop_key = []
    for key in function_list.keys():
        if function_list[key] < function_count / 20:
            drop_key.append(key)
            function_list['others'] += function_list[key]
    print(function_list)
    for key in drop_key:
        function_list.pop(key)
    function_list = {k: v for k, v in sorted(function_list.items(), key=lambda x: x[0])}
    # -------------------------------------------------------- #
    # step 2. 绘制饼图
    # -------------------------------------------------------- #
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0, bottom=0, right=0.8, top=1)
    plt.pie(x=function_list.values(), labels=function_list.keys(), colors=[colors[i] for i in function_list.keys()])
    plt.legend(bbox_to_anchor=(1.3, 0.6))
    plt.savefig('user count function{}.pdf'.format(name), dpi=1200)
    plt.show()


def plot_user_counts(data):
    '''
    绘制不同调用函数频次的用户群体的调用函数的比例（数据处理部分）
    绘制不同调用函数频次的用户分布与比例
    :param data: transactions表 Core + Sale Auction
    :return:
    '''
    print(len(data))
    user_counts = pd.DataFrame(data['transactions_from_address'].value_counts())
    # -------------------------------------------------------- #
    # step 0. 初始化不同用户群体的字典
    # -------------------------------------------------------- #
    function_list1_3 = {}
    function_list4_8 = {}
    function_list9_20 = {}
    function_list21_55 = {}
    function_list55_ = {}
    print(user_counts)
    # -------------------------------------------------------- #
    # step 1. 统计调用频次
    # -------------------------------------------------------- #
    for i in tqdm(range(len(data))):
        method = data.loc[i, 'transactions_input'][:10]
        if user_counts.loc[data.loc[i, 'transactions_from_address'], 'transactions_from_address'] <= 3:
            function_list1_3.setdefault(method, 0)
            function_list1_3[method] += 1
        elif 4 <= user_counts.loc[data.loc[i, 'transactions_from_address'], 'transactions_from_address'] <= 8:
            function_list4_8.setdefault(method, 0)
            function_list4_8[method] += 1
        elif 9 <= user_counts.loc[data.loc[i, 'transactions_from_address'], 'transactions_from_address'] <= 20:
            function_list9_20.setdefault(method, 0)
            function_list9_20[method] += 1
        elif 21 <= user_counts.loc[data.loc[i, 'transactions_from_address'], 'transactions_from_address'] <= 55:
            function_list21_55.setdefault(method, 0)
            function_list21_55[method] += 1
        else:
            function_list55_.setdefault(method, 0)
            function_list55_[method] += 1
    # -------------------------------------------------------- #
    # step 2. 绘制饼图
    # -------------------------------------------------------- #
    for i, fl in enumerate(
            [function_list1_3, function_list4_8, function_list9_20, function_list21_55, function_list55_]):
        _pie_plot_user(fl, i)

    user_counts.reset_index(inplace=True, drop=False)
    for i in range(1000):
        print(user_counts.loc[i, 'index'], end=' ')
        print(user_counts.loc[i, 'transactions_from_address'])

    # -------------------------------------------------------- #
    # step 3. 绘制小于调用频次小于100的用户分布（distplot）
    # -------------------------------------------------------- #
    user_counts1 = user_counts.loc[user_counts['transactions_from_address'] < 100]
    del data
    plt.figure(figsize=(16, 6))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    sns.kdeplot(data=user_counts1, x='transactions_from_address', fill=True)
    if ifShow:
        plt.show()
    else:
        plt.savefig('User Count kde.pdf', dpi=300)
    plt.close()
    print('\033[1;32m -- User Count kde done! --\033[0m')
    # -------------------------------------------------------- #
    # step 4. 绘制小于调用频次小于100的用户分布（kdeplot）
    # -------------------------------------------------------- #
    plt.figure(figsize=(16, 6))
    plt.subplots_adjust(left=0.12, bottom=0.17, right=0.92, top=0.95)
    sns.histplot(data=user_counts1, x='transactions_from_address', binwidth=1)
    if ifShow:
        plt.show()
    else:
        plt.savefig('User Count hist.pdf', dpi=300)
    plt.close()
    print('\033[1;32m -- User Count hist done! --\033[0m')
    # -------------------------------------------------------- #
    # step 4. 绘制不同调用频次的用户的比例
    # -------------------------------------------------------- #
    pie = {'1-3': 0, '4-8': 0, '9-20': 0, '21-55': 0, '>55': 0}
    for i in range(len(user_counts)):
        if user_counts.loc[i, 'transactions_from_address'] < 4:
            pie['1-3'] += 1
        elif user_counts.loc[i, 'transactions_from_address'] < 9:
            pie['4-8'] += 1
        elif user_counts.loc[i, 'transactions_from_address'] < 21:
            pie['9-20'] += 1
        elif user_counts.loc[i, 'transactions_from_address'] < 55:
            pie['21-55'] += 1
        else:
            pie['>55'] += 1
    print(pie)
    plt.figure(figsize=(8, 8))
    plt.subplots_adjust(left=0, bottom=0, right=0.85, top=1)
    plt.pie(x=pie.values(), labels=pie.keys(),
            colors=['#384871', '#E7BD39', '#AB4F3F', '#47855A', '#000000'],
            explode=[0.02 for i in range(5)], startangle=90)
    plt.legend(bbox_to_anchor=(1, 0.6))
    if ifShow:
        plt.show()
    else:
        plt.savefig('User Count pie.pdf', dpi=1200)
    print('\033[1;32m -- User Count pie done! --\033[0m')

    # plt.close()
    # plt.clf()
    # import scipy.stats as stats
    # import numpy as np
    # x = user_counts['transactions_from_address'].to_numpy()
    # x = np.log(x + 1)
    # stats.probplot(x, dist='norm', plot=plt)
    # plt.xlim(0, 4)
    # plt.ylim(2, 11)
    # plt.show()


def _get_kittyId(x):
    '''
    返回不同函数input data中的kittyId
    :param x: 10+64*k位16进制数
    :return:
    '''
    method1 = ['0x3d7d3f5a', '0x4ad8c938', '0x88c2a0bf', '0x454a2ab3']
    method2 = ['0xf7d8c883', '0xed60ade6']
    method3 = ['0x095ea7b3', '0xa9059cbb']
    if x[:10] in method1:
        return int(x[10:10 + 64], base=16)
    elif x[:10] in method2:
        return [int(x[10:10 + 64], base=16), int(x[10 + 64:10 + 64 * 2], base=16)]
    elif x[:10] in method3:
        return int(x[10 + 64:10 + 64 * 2], base=16)
    else:
        return False


def find_bid_address(sale_inform, data_sale):
    '''
    查询createSaleAuction是否成功，如成功，则返回买家地址
    :param sale_inform: createSaleAuction，一行
    :param data_sale: SaleAuction，bid函数，整表
    :return: bid_address
    '''
    # -------------------------------------------------------- #
    # step 1. 通过猫的Id找到候选者（该猫对应的所有bid记录）
    # -------------------------------------------------------- #
    method = '0x454a2ab3'
    hex_kittyId = sale_inform['transactions_input'][10: 10 + 64]
    # duration = int(sale_inform['transactions_input'][10 + 64 * 3:], base=16)
    timestamp_start = sale_inform['transactions_block_timestamp']
    # timestamp_end = int(timestamp_start) + duration
    candidates = data_sale.loc[data_sale['transactions_input'].str.startswith(method + hex_kittyId)]
    # -------------------------------------------------------- #
    # step 2. 查找拍卖之前的第一次createSaleAuction
    # -------------------------------------------------------- #
    for i in candidates.index:
        if timestamp_start <= int(candidates.loc[i, 'transactions_block_timestamp']):
            # print(1, candidates.loc[i, 'transactions_from_address'])
            return candidates.loc[i, 'transactions_from_address']
    # if len(candidates) != 0:
    #     print(candidates)
    return None


def plot_user_behaviour(data_core, data_sale, user_address):
    '''
    绘制用户行为图
    :param data_core: transactions表 Core合约
    :param data_sale: transactions表 Sale Auction合约
    :param user_address: 中心用户/绘制哪个用户的行为
    :return:
    '''
    # user_address = '0x2bccf86a7315190d974330a75cb31c3f66c1175d'
    # user_address = '0x0429c8d18b916dffa9d3ac0bc56d34d9014456ef'
    # user_address = '0x75771dedde9707fbb78d9f0dbdc8a4d4e7784794'
    # plt.figure(figsize=(16, 16))
    graph = nx.MultiDiGraph()
    node_user = [user_address]
    node_kitty = []
    plt.figure(figsize=(16, 12))
    # -------------------------------------------------------- #
    # step 1. 买了哪些猫？ bid
    # -------------------------------------------------------- #
    edge_bid = []
    user_bid = data_sale.loc[data_sale['transactions_input'].str.startswith('0x454a2ab3')]
    user_bid = user_bid.loc[data_sale['transactions_from_address'] == user_address]
    user_bid.reset_index(inplace=True, drop=True)
    for i in range(len(user_bid)):
        kittyId = _get_kittyId(user_bid.loc[i, 'transactions_input'])
        node_kitty.append(kittyId)
        edge_bid.append((kittyId, user_address))

    # -------------------------------------------------------- #
    # step 2. 买了哪些猫的交配权？
    # -------------------------------------------------------- #
    edge_bidOnS = []
    edge_bidOnS_k2k = []

    user_core = data_core.loc[data_core['transactions_from_address'] == user_address]
    user_bidOnS = user_core.loc[user_core['transactions_input'].str.startswith('0xed60ade6')]
    user_bidOnS.reset_index(inplace=True, drop=True)
    for i in range(len(user_bidOnS)):
        sireId, matronId = _get_kittyId(user_bidOnS.loc[i, 'transactions_input'])
        node_kitty.append(sireId)
        node_kitty.append(matronId)
        edge_bidOnS.append((matronId, user_address))
        edge_bidOnS_k2k.append((sireId, matronId))

    # -------------------------------------------------------- #
    # step 3. 创建了哪些拍卖？ 哪些拍卖成功了？
    # -------------------------------------------------------- #
    edge_allSale = []
    edge_transfer = []

    allSale = user_core.loc[user_core['transactions_input'].str.startswith('0x3d7d3f5a')]
    allSale.reset_index(inplace=True, drop=True)

    for i in tqdm(range(len(allSale))):
        kittyId = _get_kittyId(allSale.loc[i, 'transactions_input'])
        bid_address = find_bid_address(allSale.loc[i, :], data_sale)
        node_kitty.append(kittyId)
        edge_allSale.append((user_address, kittyId))
        if not bid_address is None:
            # print(bid_address)
            node_user.append(bid_address)
            edge_transfer.append((kittyId, bid_address))

    # -------------------------------------------------------- #
    # step 4. 自家猫咋生的？
    # -------------------------------------------------------- #
    edge_bWA = []
    bWA = user_core.loc[user_core['transactions_input'].str.startswith('0xf7d8c883')]
    bWA.reset_index(inplace=True, drop=True)
    print(bWA)
    for i in range(len(bWA)):
        sireId, matronId = _get_kittyId(bWA.loc[i, 'transactions_input'])
        node_kitty.append(sireId)
        node_kitty.append(matronId)
        edge_bidOnS.append((matronId, user_address))
        edge_bWA.append((sireId, matronId))

    # -------------------------------------------------------- #
    # step 5. 添加node 和节点
    # -------------------------------------------------------- #
    node_kitty = list(set(node_kitty))
    graph.add_nodes_from(node_user)
    graph.add_nodes_from(node_kitty)
    graph.add_edges_from(edge_bid)

    graph.add_edges_from(edge_bidOnS)
    graph.add_edges_from(edge_bidOnS_k2k)
    graph.add_edges_from(edge_transfer)

    graph.add_edges_from(edge_allSale)

    graph.add_edges_from(edge_bWA)

    # -------------------------------------------------------- #
    # step 6. 画图
    # -------------------------------------------------------- #
    pos = nx.spring_layout(graph)
    # pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, nodelist=node_user, pos=pos, node_size=200, node_color=color_list[0])
    nx.draw_networkx_nodes(graph, nodelist=node_kitty, pos=pos, node_size=5, node_color=color_list[1])
    nx.draw_networkx_edges(graph, edgelist=edge_bid, pos=pos, width=4, edge_color=color_list[2])

    nx.draw_networkx_edges(graph, edgelist=edge_bidOnS, pos=pos, width=2, edge_color=color_list[3])
    nx.draw_networkx_edges(graph, edgelist=edge_bidOnS_k2k, pos=pos, width=1, edge_color=color_list[4])

    nx.draw_networkx_edges(graph, edgelist=edge_allSale, pos=pos, width=0.1, edge_color=color_list[5])
    nx.draw_networkx_edges(graph, edgelist=edge_transfer, pos=pos, width=0.1, edge_color=color_list[6])
    nx.draw_networkx_edges(graph, edgelist=edge_bWA, pos=pos, width=0.1, edge_color=color_list[7])

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=color_list[0], lw=4),
                    Line2D([0], [0], color=color_list[1], lw=4),
                    Line2D([0], [0], color=color_list[2], lw=4),
                    Line2D([0], [0], color=color_list[3], lw=4),
                    Line2D([0], [0], color=color_list[4], lw=4),
                    Line2D([0], [0], color=color_list[5], lw=4),
                    Line2D([0], [0], color=color_list[6], lw=4),
                    Line2D([0], [0], color=color_list[7], lw=4)]

    plt.legend(custom_lines, ['Address', 'Kitty', 'Bid', 'Bid On Siring',
                              'Bid On Siring k2k', 'Create Sale', 'Bid Success', 'Breed With Auto'])
    # -------------------------------------------------------- #
    # step 7. 输出整张图
    # -------------------------------------------------------- #
    if ifShow:
        pass
    else:
        plt.savefig('network1.pdf', dpi=1200)
    print('\033[1;32m -- network1 done! -- \033[0m')
    # -------------------------------------------------------- #
    # step 8. 放大再输出一遍
    # -------------------------------------------------------- #
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    print(xlim, ylim)
    plt.xlim(xlim[0] + 0.46 * (xlim[1] - xlim[0]), xlim[0] + 0.54 * (xlim[1] - xlim[0]))
    plt.ylim(ylim[0] + 0.46 * (ylim[1] - ylim[0]), ylim[0] + 0.54 * (ylim[1] - ylim[0]))
    if ifShow:
        plt.show()
    else:
        plt.savefig('network2.pdf', dpi=1200)
    print('\033[1;32m -- network2 done! -- \033[0m')


def cat_relationship():
    '''
    绘制猫的五度好友
    :return:
    '''
    kitty_data = pd.read_csv('../data/kitty_table.csv', index_col='kittyId')
    plt.figure(figsize=(20, 16))
    graph = nx.MultiDiGraph()
    node_list = []
    edge_list = []
    # -------------------------------------------------------- #
    # step 0. 指定猫的ID
    # -------------------------------------------------------- #
    chosenkittyId = 202302

    node_list.append(chosenkittyId)
    # -------------------------------------------------------- #
    # step 1. 循环五次，每次寻找nodelist中所有猫的父亲、母亲和孩子
    # -------------------------------------------------------- #
    for i in range(5):
        for j in tqdm(range(len(node_list))):
            kittyId = node_list[j]
            matronId = kitty_data.loc[kittyId, 'matronId']
            sireId = kitty_data.loc[kittyId, 'sireId']
            node_list.append(matronId)
            node_list.append(sireId)
            edge_list.append((matronId, kittyId))
            edge_list.append((sireId, kittyId))
            # -------------------------------------------------------- #
            # step 2. 找到表中父亲Id或母亲Id为该猫的行，其kittyId即该猫的孩子
            # -------------------------------------------------------- #
            child1 = kitty_data.loc[kitty_data['matronId'] == kittyId]
            child2 = kitty_data.loc[kitty_data['sireId'] == kittyId]
            children = pd.concat([child1, child2], axis=0)
            for childId in children.index:
                node_list.append(childId)
                edge_list.append((kittyId, childId))
    # -------------------------------------------------------- #
    # step 3. 添加节点与边
    # -------------------------------------------------------- #
    graph.add_nodes_from(node_list)
    graph.add_edges_from(edge_list)
    # -------------------------------------------------------- #
    # step 4. 画图
    # -------------------------------------------------------- #
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, nodelist=node_list, pos=pos, node_size=20, node_color=color_list[0])
    nx.draw_networkx_edges(graph, edgelist=edge_list, pos=pos, width=1, edge_color=color_list[1])

    plt.savefig('cat relationship.pdf', dpi=1200)
    print('\033[1;32m -- cat relationship done! -- \033[0m')


if __name__ == '__main__':
    color_list = ['#81b29a', '#f2cc8f', '#e07a5f', '#e0b1cb',
                  '#9b2226', '#1e6091', '#809bce', '#d9ed92',
                  '#b8c0ff']
    ifShow = False  # 用于展示，不保存图片
    # -------------------------------------------------------- #
    # 第一部分 count
    # -------------------------------------------------------- #
    data_path = '../data/Core.csv'
    data = pd.read_csv(data_path)
    print(data.columns)
    plot_cSaA(data)
    plot_cSiA(data)
    plot_gB(data)
    plot_bWA(data)
    plot_bid_sir(data)

    data_path = '../data/sale auction.csv'
    data = pd.read_csv(data_path)
    print(data.columns)
    plot_bid(data)
    plot_createMinusBid()

    # -------------------------------------------------------- #
    # 第二部分 price
    # -------------------------------------------------------- #
    data_path = '../data/Core.csv'
    data = pd.read_csv(data_path)
    print(data.columns)
    # 生成 日期 - price 字典
    price_data = pd.read_csv('../data/eth price.csv')
    price = {}
    for i in range(len(price_data)):
        price[price_data.loc[i, 'create_date']] = price_data.loc[i, 'eth_price']
    price['2023-02-07'] = 1670.40
    plot_price(data, price)

    data_path = '../data/sale auction.csv'
    data = pd.read_csv(data_path)
    print(data.columns)
    plot_bid_price(data, price)

    # 以太币的增长幅度
    plot_eth()

    # -------------------------------------------------------- #
    # 第三部分 users
    # -------------------------------------------------------- #
    data_path = '../data/Core.csv'
    data = pd.read_csv(data_path)
    data_path = '../data/sale auction.csv'
    df = pd.read_csv(data_path)

    data = pd.concat([data, df])
    data.reset_index(inplace=True)

    plot_user_counts(data)

    # -------------------------------------------------------- #
    # 第四部分 user behaviour
    # -------------------------------------------------------- #
    data_path = '../data/Core.csv'
    data = pd.read_csv(data_path)
    # data = data.loc[data['transactions_day'].str.startswith('2018-01')]
    data_path = '../data/sale auction.csv'
    df = pd.read_csv(data_path)
    # df = df.loc[df['transactions_day'].str.startswith('2018-01')]
    # user_address = '0x75771dedde9707fbb78d9f0dbdc8a4d4e7784794'
    user_address = '0x85a243ceb8539884f0aa935408256c7f37c79ad3'
    plot_user_behaviour(data, df, user_address)
    # -------------------------------------------------------- #
    # 第五部分 cat_relationship
    # -------------------------------------------------------- #
    cat_relationship()
