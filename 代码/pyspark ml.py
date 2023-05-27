import pandas as pd
from tqdm import tqdm

# pyspark
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import col, log, pow
from pyspark.sql import SparkSession

# plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

kai = [str(i) for i in range(1, 10)]
kai.extend([chr(i) for i in range(97, 121) if i != 108])


def _decode_genes(x):
    '''
    将基因编码解析为kai编码
    :param x: 64位16进制数
    :return:
    '''
    # ----------------------------------------------- #
    # step 1. 从log data中截取gene对应的部分
    # ----------------------------------------------- #
    x = x[2 + 64 * 4:2 + 64 * 5]
    # ----------------------------------------------- #
    # step 2. 转换为2进制
    # ----------------------------------------------- #
    gene_bin = ''.join(str(bin(int(i, base=16))).replace('0b', '').zfill(4) for i in x)
    # ----------------------------------------------- #
    # step 3. 去掉前16位，每5位一组，转换为kai编码
    # ----------------------------------------------- #
    genes = ''
    for i in range(16, len(gene_bin), 5):
        genes += kai[int(gene_bin[i:i + 5], base=2)]
    return genes


def get_generation(data):
    '''
    递归方法获取代数（第一代、第二代）
    :param data: kitty table
    :return:
    '''
    data['generation'] = None
    data['pregnant'] = ''
    # ----------------------------------------------- #
    # step 1. 初代猫的父母地址为0，将初代猫设置为0代
    # ----------------------------------------------- #
    for kittyId in data.index:
        if not (data.loc[kittyId, 'matronId'] or data.loc[kittyId, 'sireId']):
            data.loc[kittyId, 'generation'] = 0
    # ----------------------------------------------- #
    # step 2. 递归方法获取代数
    # ----------------------------------------------- #
    for kittyId in tqdm(data.index):
        matronId = data.loc[kittyId, 'matronId']
        data.loc[matronId, 'pregnant'] += ' ' + str(data.loc[kittyId, 'birthBlock'])
        if pd.isna(data.loc[kittyId, 'generation']):
            data = _get_generation(kittyId, data)
    return data


def _get_generation(kittyId, data):
    '''
    递归，如果当前的父母generation为None，则先填写父母的generation
    :param kittyId: 当前猫的Id
    :param data: 当前状态的kitty table
    :return:
    '''
    matronId = data.loc[kittyId, 'matronId']
    sireId = data.loc[kittyId, 'matronId']
    matronGeneration = data.loc[matronId, 'generation']
    sireGeneration = data.loc[sireId, 'generation']
    if matronGeneration is None:
        data = _get_generation(matronId, data)
        print(data.loc[matronId, 'generation'])
    if sireGeneration is None:
        data = _get_generation(sireId, data)
        print(data.loc[matronId, 'generation'])
    data.loc[kittyId, 'generation'] = max(matronGeneration, sireGeneration) + 1

    return data


def get_kitty_data():
    '''
    整合猫的信息
    :return: kitty table
    '''
    pd.options.display.max_columns = None
    log = pd.read_csv('../data/log.csv')
    print('\033[1;32m -- log data loaded-- \033[0m')

    # ----------------------------------------------------------- #
    # step 1. 建立猫的表
    # ----------------------------------------------------------- #
    birth_topic = '0x0a5311bd2a6608f08a180df2ee7c5946819a649b204b554bb8e39825b2c50ad5'
    birth_inform = log.loc[log['logs_topics'] == birth_topic]
    birth_inform = birth_inform.drop_duplicates()
    kitty_table = pd.DataFrame()
    # kitty_table['owner'] = birth_inform['logs_data'].map(lambda x: '0x' + x[2:2 + 64].lstrip('0'))
    # print('\033[1;32m -- get owner done! -- \033[0m')
    kitty_table['kittyId'] = birth_inform['logs_data'].map(lambda x: int(x[2 + 64:2 + 64 * 2], base=16))
    print('\033[1;32m -- get kittyId done! -- \033[0m')
    kitty_table['matronId'] = birth_inform['logs_data'].map(lambda x: int(x[2 + 64 * 2:2 + 64 * 3], base=16))
    print('\033[1;32m -- get matronId done! -- \033[0m')
    kitty_table['sireId'] = birth_inform['logs_data'].map(lambda x: int(x[2 + 64 * 3:2 + 64 * 4], base=16))
    print('\033[1;32m -- get sireId done! -- \033[0m')
    kitty_table['genes'] = birth_inform['logs_data'].map(_decode_genes)
    print('\033[1;32m -- decode gene done! -- \033[0m')
    kitty_table['birthBlock'] = birth_inform['logs_block_number']
    kitty_table.set_index(kitty_table['kittyId'], inplace=True, drop=True)
    kitty_table.drop(labels='kittyId', axis=1, inplace=True)
    kitty_table.sort_index(inplace=True)
    print(kitty_table)
    kitty_table = get_generation(kitty_table)
    print(kitty_table)
    kitty_table.to_csv('../data/kitty_table.csv', index=True)


def get_market_data():
    '''
    获取市场数据，包括各种函数的调用频次
    :return:
    '''
    core = pd.read_csv('../data/Core.csv')
    data = pd.DataFrame()
    for i in ['0x3d7d3f5a', '0x4ad8c938', '0x88c2a0bf', '0xf7d8c883', '0xed60ade6']:
        df = core.loc[core['transactions_input'].str.startswith(i)]
        df = pd.DataFrame(df['transactions_day'].value_counts())
        df.columns = [i]
        data = pd.concat([data, df], axis=1)
    saleAuction = pd.read_csv('../data/sale auction.csv')
    df = saleAuction.loc[saleAuction['transactions_input'].str.startswith('0x454a2ab3')]
    df = pd.DataFrame(df['transactions_day'].value_counts())
    print(df)
    df.columns = ['0x454a2ab3']
    data = pd.concat([data, df], axis=1)

    data.sort_index(ascending=True, inplace=True)
    data = data.fillna(0)
    data.to_csv('../data/market.csv')


def load_data():
    '''
    加载数据 包括 Core SaleAuction price market kitty_table
    :return:
    '''
    core = pd.read_csv('../data/Core.csv')
    core = core.loc[core['transactions_day'].str.startswith(('2017', '2018'))]
    print('\033[1;32m -- core data loaded -- \033[0m')
    saleauction = pd.read_csv('../data/sale auction.csv')
    saleauction = saleauction.loc[saleauction['transactions_day'].str.startswith('2018')]
    print('\033[1;32m -- sale auction data loaded -- \033[0m')
    kitty = pd.read_csv('../data/kitty_table.csv', index_col='kittyId')
    print('\033[1;32m -- kitty data loaded -- \033[0m')
    price_data = pd.read_csv('../data/eth price.csv')
    print('\033[1;32m -- price loaded -- \033[0m')
    price = {}
    for i in range(len(price_data)):
        price[price_data.loc[i, 'create_date']] = price_data.loc[i, 'eth_price']
    price['2023-02-07'] = 1670.40
    market = pd.read_csv('../data/market.csv', index_col='Unnamed: 0')
    print('\033[1;32m -- market loaded -- \033[0m')

    return core, saleauction, kitty, price, market


def find_createSale(bid_inform, createSale, kitty_data):
    '''
    通过bid信息查询对应的createSaleAcution信息
    :param bid_inform: bid信息
    :param createSale: 所有create Sale Auction的表
    :param kitty_data: 猫的表，为了判断这只猫是不是初代猫
    :return:
    '''
    # method = '0x3d7d3f5a'
    # hex_kittyId = bid_inform['transactions_input'][10: 10 + 64]
    # candidates = createSale.loc[createSale['transactions_input'].str.startswith(method + hex_kittyId)]
    # timestamp_bid = bid_inform['transactions_block_timestamp']
    # candidates = candidates.sort_values('transactions_block_timestamp', ascending=False)
    kittyId = bid_inform['kittyId']
    timestamp_bid = bid_inform['transactions_block_timestamp']
    # 第一种 情况 createSaleAuction里有这只猫
    try:
        candidates = createSale.loc[kittyId, :, :]
        candidates = candidates.sort_values('transactions_block_timestamp', ascending=False)
        for i in candidates.index:
            if timestamp_bid >= int(candidates.loc[i, 'transactions_block_timestamp']):
                # duration = int(candidates.loc[i, 'transactions_input'][10 + 64 * 3:], base=16)
                return candidates.loc[i, 'transactions_input']
    except:
        pass
    # 第二种情况： 这只猫是gen0:
    if kitty_data['matronId'] == 0 and kitty_data['sireId'] == 0:
        hex_kittyId = str(hex(kittyId))[2:].zfill(64)
        createSaleInput = '0xgen0auct{}000000000000000000000000000000000000000000000000138cc38d9c97bca000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000015180'.format(
            hex_kittyId)
        return createSaleInput
    print(bid_inform['transactions_hash'])  # 找不到的 都是 bid fail的
    # if len(candidates) != 0:
    #     print(candidates)
    return None


def count_pregnant(bid_inform, kitty_data):
    '''
    统计bid之前的生育次数
    :param bid_inform: 拍卖信息（主要获取拍卖区块）
    :param kitty_data: 猫的表（主要获取pregnant列表）
    :return:
    '''
    bid_num = bid_inform['transactions_block_number']
    if pd.isna(kitty_data['pregnant']):
        return 0
    else:
        pregnant_list = [int(i) for i in kitty_data['pregnant'].split()]
        count = len([i for i in pregnant_list if i < bid_num])
        return count


def get_sale_data():
    '''
    数据整合，获得所有拍卖成功的销售数据
    :return:
    '''
    core, saleauction, kitty, price, market = load_data()
    sale_data = saleauction.loc[saleauction['transactions_input'].str.startswith('0x454a2ab3')]
    sale_data = sale_data[['transactions_hash',
                           'transactions_block_number',
                           'transactions_transaction_index',
                           'transactions_value',
                           'transactions_gas_price',
                           'transactions_input',
                           'transactions_day',
                           'transactions_block_timestamp']]
    # 加入kittyId 加快索引
    sale_data['kittyId'] = sale_data['transactions_input'].map(lambda x: int(x[10:10 + 64], base=16))
    sale_data['eth_price'] = sale_data['transactions_day'].map(lambda x: price[x])
    sale_data['createSaleInput'] = None
    sale_data['pregnant_count'] = None
    sale_data['generation'] = None
    sale_data['genes'] = None
    for i in market.columns:
        sale_data[i] = None
    # 找create Sale Auction
    sale_data.reset_index(inplace=True, drop=True)
    # 一种是Create Sale Auction 一种是 create Gen0 Auction
    creatSale = core.loc[core['transactions_input'].str.startswith('0x3d7d3f5a')]
    creatSale = creatSale.loc[creatSale['transactions_input'].str.len() == 266]
    creatSale['kittyId'] = creatSale['transactions_input'].map(lambda x: int(x[10:10 + 64], base=16))
    creatSale['index'] = creatSale.index
    creatSale.set_index(['kittyId', 'index'], inplace=True)

    for i in tqdm(range(len(sale_data))):
        # for i in tqdm(range(10000)):
        try:
            kit = kitty.loc[sale_data.loc[i, 'kittyId']]
        except:
            print(sale_data.loc[i, 'kittyId'])
            continue
        sale_data.loc[i, 'createSaleInput'] = find_createSale(sale_data.loc[i, :], creatSale, kit)
        sale_data.loc[i, 'pregnant_count'] = count_pregnant(sale_data.loc[i, :], kit)
        sale_data.loc[i, 'generation'] = kit['generation']
        sale_data.loc[i, 'genes'] = kit['genes']
        for j in market.columns:
            sale_data.loc[i, j] = market.loc[sale_data.loc[i, 'transactions_day'], j]

    sale_data.dropna(axis=0, how='any', subset=['createSaleInput'], inplace=True)
    sale_data.reset_index(inplace=True, drop=True)

    sale_data['duration'] = sale_data['createSaleInput'].map(lambda x: int(x[10 + 64 * 3:], base=16))
    sale_data['start_price'] = sale_data['createSaleInput'].map(lambda x: int(x[10 + 64:10 + 64 * 2], base=16))
    sale_data['end_price'] = sale_data['createSaleInput'].map(lambda x: int(x[10 + 64 * 2:10 + 64 * 3], base=16))

    # 将基因转换为类别变量，两种转换方法：1. 只转换显性形状 2. 转换全部基因
    # for i in range(12):
    #     sale_data['gene{}'.format(i + 1)] = sale_data['genes'].map(lambda x: x[i * 4 + 3])

    for i in range(48):
        sale_data['gene{}'.format(i + 1)] = sale_data['genes'].map(lambda x: x[i])

    print(sale_data)
    sale_data.to_csv('../data/sale_data.csv', index=False)


def lr():
    '''
    LinearRegression
    预测猫的拍卖价格
    :return:
    '''
    spark = SparkSession.builder.appName('lin_reg').getOrCreate()
    data = spark.read.csv('../data/sale_data.csv', inferSchema=True, header=True)
    # ----------------------------------------------------------- #
    # step 1. 去除无关变量
    # ----------------------------------------------------------- #
    data = data.drop('transactions_hash', 'transactions_block_number', 'transactions_transaction_index',
                     'transactions_input', 'transactions_day', 'createSaleInput', 'genes',
                     'start_price', 'end_price')  # 'kittyId'
    data.printSchema()
    print(data.count())
    # for i in range(12):
    #     stingIndexder = StringIndexer(inputCol='gene{}'.format(i + 1), outputCol='gene_{}'.format(i + 1))
    #     data = stingIndexder.fit(data).transform(data).drop('gene{}'.format(i + 1))
    #     onehotencoder = OneHotEncoder(inputCol='gene_{}'.format(i + 1), outputCol='gene{}'.format(i + 1))
    #     data = onehotencoder.fit(data).transform(data).drop('gene_{}'.format(i + 1))
    # ----------------------------------------------------------- #
    # step 2. 将类别变量转换为独热编码
    # ----------------------------------------------------------- #
    for i in range(48):
        stingIndexder = StringIndexer(inputCol='gene{}'.format(i + 1), outputCol='gene_{}'.format(i + 1))
        data = stingIndexder.fit(data).transform(data).drop('gene{}'.format(i + 1))
        onehotencoder = OneHotEncoder(inputCol='gene_{}'.format(i + 1), outputCol='gene{}'.format(i + 1))
        data = onehotencoder.fit(data).transform(data).drop('gene_{}'.format(i + 1))

    onehotencoder = OneHotEncoder(inputCol='generation', outputCol='generation_onehot')
    data = onehotencoder.fit(data).transform(data).drop('generation')
    # ----------------------------------------------------------- #
    # step 3. 生成bid price， 删除transactoins value
    # ----------------------------------------------------------- #
    data = data.withColumn('bid_price', log(col('transactions_value') * col('eth_price') * 10 ** -18 + 1)).drop(
        'transactions_value')
    # ----------------------------------------------------------- #
    # step 4. 对数处理
    # ----------------------------------------------------------- #
    for col_name in ['duration', 'transactions_gas_price', 'pregnant_count',
                     'eth_price', '0x3d7d3f5a', '0x4ad8c938', '0x88c2a0bf',
                     '0xf7d8c883', '0xed60ade6', '0x454a2ab3'
                     ]:
        data = data.withColumn('log {}'.format(col_name), log(col(col_name) + 1)).drop(col_name)
    # ----------------------------------------------------------- #
    # step 5. 添加时间的高阶项
    # ----------------------------------------------------------- #
    data = data.withColumn('transactions_block_timestamp2', pow(data['transactions_block_timestamp'], 2))
    data = data.withColumn('transactions_block_timestamp3', pow(data['transactions_block_timestamp'], 3))
    # data = data.withColumn('log start_price', log(col('start_price') + 1)).drop('start_price')
    # data = data.withColumn('log end_price', log(col('end_price') + 1)).drop('end_price')

    data.printSchema()
    data.show(5, False)
    # ----------------------------------------------------------- #
    # step 6. 转换为向量
    # ----------------------------------------------------------- #
    assembler = VectorAssembler(inputCols=[i for i in data.columns if i != 'bid_price'], outputCol="features")
    data = assembler.transform(data)
    data = data.select('features', 'bid_price')
    data.printSchema()
    data.show(5)
    data.select('features').show(5, False)

    # scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True, withMean=True)
    # data = scaler.fit(data).transform(data)
    # lin_Reg = LinearRegression(labelCol='bid_price', featuresCol='scaledFeatures')
    # ----------------------------------------------------------- #
    # step 7. 线性回归
    # ----------------------------------------------------------- #
    lin_Reg = LinearRegression(labelCol='bid_price', featuresCol='features', solver="normal")
    lr_model = lin_Reg.fit(data)
    # ----------------------------------------------------------- #
    # step 8. 输出结果
    # ----------------------------------------------------------- #
    print(lr_model.intercept)
    print(lr_model.coefficients)
    # ----------------------------------------------------------- #
    # step 9. 模型评估
    # ----------------------------------------------------------- #
    training_predictions = lr_model.evaluate(data)

    print(f'MSE:{training_predictions.meanSquaredError}')
    print(f'r2:{training_predictions.r2}')
    # ----------------------------------------------------------- #
    # step 10. 参数绘图
    # ----------------------------------------------------------- #
    plot_coff(coef=lr_model.coefficients)
    # summary = lr_model.summary
    # print(summary.pValues)


def _plot(data, x):
    plt.figure(figsize=(8, 4))
    plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95, hspace=0.36)
    ax1 = plt.subplot(2, 1, 1)
    # sns.displot(data=data, x=x, kind='kde', fill=True, ax=ax1)
    sns.kdeplot(data=data, x=x, fill=True, ax=ax1)
    ax2 = plt.subplot(2, 1, 2)
    data['log ' + x] = data[x].map(lambda x: np.log(x + 1))
    # sns.displot(data=data, x=x, kind='kde', fill=True, ax=ax2)
    sns.kdeplot(data=data, x='log ' + x, fill=True, ax=ax2)
    plt.savefig('variable ' + x + '.pdf', dpi=1200)

    # plt.show()


def plot_variables():
    '''
    查看各个变量的分布情况
    :return:
    '''
    data = pd.read_csv('../data/sale_data.csv')
    data = data.drop(['transactions_hash', 'transactions_block_number', 'transactions_transaction_index',
                      'transactions_input', 'transactions_day', 'createSaleInput', 'genes'], axis=1)
    for i in range(12):
        df_dummies = pd.get_dummies(data['gene{}'.format(i + 1)], drop_first=True, prefix='gene{}'.format(i + 1))
        data.drop('gene{}'.format(i + 1), axis=1, inplace=True)
        data = pd.concat([data, df_dummies], axis=1)
    data['bid_price'] = data['transactions_value'].astype(float) * data['eth_price'] * 10 ** -18
    data.to_csv('../data/temp.csv', index=False)
    print(data.columns)
    data['transactions_value'] = data['transactions_value'].astype(float)
    data['start_price'] = data['start_price'].astype(float)
    data['end_price'] = data['end_price'].astype(float)
    _plot(data, 'transactions_value')
    _plot(data, 'bid_price')
    _plot(data, 'transactions_gas_price')
    _plot(data, 'eth_price')
    _plot(data, 'duration')
    _plot(data, 'pregnant_count')
    _plot(data, 'generation')
    _plot(data, 'start_price')
    _plot(data, 'end_price')


def plot_coff(coef):
    '''
    以热力图的形式展现模型参数
    :param coef: 模型参数
    :return:
    '''
    import numpy as np
    gene_nums = [27, 30, 31, 31, 24, 23, 25, 27, 29, 30,
                 31, 31, 30, 31, 31, 31, 30, 31, 31, 31,
                 27, 31, 31, 31, 29, 30, 31, 31, 29, 31,
                 31, 31, 30, 31, 31, 31, 28, 30, 31, 31,
                 29, 31, 31, 31, 29, 30, 31, 31]
    print(len(coef))
    print(sum(gene_nums))
    gene_coef = [[0 for i in range(31)] for j in range(48)]
    cnt = 2
    for i, num in enumerate(gene_nums):
        for j in range(num):
            gene_coef[i][j] = coef[cnt]
            cnt += 1
    print(coef[cnt:cnt + 22])
    gene_coef = np.array(gene_coef)
    plt.figure(figsize=(8, 8))
    sns.heatmap(gene_coef, vmax=1, vmin=-1,
                cmap=sns.diverging_palette(h_neg=250, h_pos=12, s=50, l=50, as_cmap=True),
                square=True, cbar=True, xticklabels=False, yticklabels=False)
    plt.savefig('gene coef.pdf', dpi=1200)
    plt.show()

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 3))
    sns.lineplot(x=range(22), y=coef[cnt:cnt + 22])
    plt.savefig('generation coef.pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    # get_kitty_data()
    # get_market_data()
    get_sale_data()
    lr()
    # plot_variables()
