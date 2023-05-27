import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import win32con
import win32api
import pyperclip
import os
import pandas as pd
import numpy as np

key_map = {
    "0": 49, "1": 50, "2": 51, "3": 52, "4": 53, "5": 54, "6": 55, "7": 56, "8": 57, "9": 58,
    'F1': 112, 'F2': 113, 'F3': 114, 'F4': 115, 'F5': 116, 'F6': 117, 'F7': 118, 'F8': 119,
    'F9': 120, 'F10': 121, 'F11': 122, 'F12': 123, 'F13': 124, 'F14': 125, 'F15': 126, 'F16': 127,
    "A": 65, "B": 66, "C": 67, "D": 68, "E": 69, "F": 70, "G": 71, "H": 72, "I": 73, "J": 74,
    "K": 75, "L": 76, "M": 77, "N": 78, "O": 79, "P": 80, "Q": 81, "R": 82, "S": 83, "T": 84,
    "U": 85, "V": 86, "W": 87, "X": 88, "Y": 89, "Z": 90,
    'BACKSPACE': 8, 'TAB': 9, 'TABLE': 9, 'CLEAR': 12,
    'ENTER': 13, 'SHIFT': 16, 'CTRL': 17,
    'CONTROL': 17, 'ALT': 18, 'ALTER': 18, 'PAUSE': 19, 'BREAK': 19, 'CAPSLK': 20, 'CAPSLOCK': 20, 'ESC': 27,
    'SPACE': 32, 'SPACEBAR': 32, 'PGUP': 33, 'PAGEUP': 33, 'PGDN': 34, 'PAGEDOWN': 34, 'END': 35, 'HOME': 36,
    'LEFT': 37, 'UP': 38, 'RIGHT': 39, 'DOWN': 40, 'SELECT': 41, 'PRTSC': 42, 'PRINTSCREEN': 42, 'SYSRQ': 42,
    'SYSTEMREQUEST': 42, 'EXECUTE': 43, 'SNAPSHOT': 44, 'INSERT': 45, 'DELETE': 46, 'HELP': 47, 'WIN': 91,
    'WINDOWS': 91, 'NMLK': 144,
    'NUMLK': 144, 'NUMLOCK': 144, 'SCRLK': 145,
    '[': 219, ']': 221, '+': 107, '-': 109}


def press_key(key_code):
    """
        函数功能：按下按键
        参   数：key:按键值
    """
    win32api.keybd_event(key_code, win32api.MapVirtualKey(key_code, 0), 0, 0)


def release_key(key_code):
    """
        函数功能：抬起按键
        参   数：key:按键值
    """
    win32api.keybd_event(key_code, win32api.MapVirtualKey(key_code, 0), win32con.KEYEVENTF_KEYUP, 0)


def connect():
    option = webdriver.ChromeOptions()
    option.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    chromedriver_path = r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'
    driver = webdriver.Chrome(options=option, executable_path=chromedriver_path)
    return driver


def get_data(driver, date):
    driver.find_element(By.XPATH,
                        '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[1]/div[1]/button[1]').click()
    driver.find_element(By.XPATH,
                        '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[2]/div[6]/div[1]/div/div/div/div[5]/div/pre').click()
    press_key(key_map["CONTROL"])
    press_key(key_map["V"])
    release_key(key_map["CONTROL"])
    release_key(key_map["V"])
    time.sleep(2)

    driver.find_element(By.XPATH,
                        '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[3]/button[1]').click()

    while True:
        try:
            driver.find_element(By.XPATH,
                                '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[3]/button[2]').click()
            time.sleep(2)
            driver.find_element(By.XPATH,
                                '/html/body/div[@class="el-dialog__wrapper queryDialogNewQuery"]/div/div[2]/div/form/div[1]/div[1]/div/div/input').send_keys(
                date)
            time.sleep(2)
            break
        except:
            time.sleep(2)

    # time.sleep(7)
    # # 保存
    #
    # driver.find_element(By.XPATH,
    #                     '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[3]/button[2]').click()
    # time.sleep(2)
    # # 保存，输入名称
    # try:
    #     driver.find_element(By.XPATH, '/html/body/div[@class="el-dialog__wrapper queryDialogNewQuery"]/div/div[2]/div/form/div[1]/div[1]/div/div/input').send_keys(
    #         date)
    # except:
    #     time.sleep(20)
    #     driver.find_element(By.XPATH,
    #                         '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[3]/button[2]').click()
    #     time.sleep(2)
    #     driver.find_element(By.XPATH, '/html/body/div[@class="el-dialog__wrapper queryDialogNewQuery"]/div/div[2]/div/form/div[1]/div[1]/div/div/input').send_keys(date)
    # time.sleep(2)
    # 保存
    driver.find_element(By.XPATH,
                        '/html/body/div[@class="el-dialog__wrapper queryDialogNewQuery"]/div/div[2]/div/form/div[2]/button').click()
    time.sleep(4)
    # 下载
    for i in range(10):
        try:
            driver.find_element(By.XPATH,
                                '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[2]/div/div[2]/button[2]').click()
            time.sleep(4)
            break
        except:
            time.sleep(4)
            if i == 9:
                raise Exception("line 110")

    time.sleep(2)


def get_core_data(start_date):
    sql_base = 'SELECT * FROM nifi_ethereum.transactions WHERE to_address = "0x06012c8cf97bead5deae237070f9587f8e7a266d" AND '
    date = datetime.strptime(start_date, "%Y-%m-%d")
    n = 365 * 10
    for i in range(n):
        day = date + timedelta(days=i)
        if day.strftime("%Y-%m-%d") == '2023-01-01':
            break
        sql = sql_base + 'day=\"' + day.strftime("%Y-%m-%d") + '\"'
        print(sql)
        pyperclip.copy(sql)

        get_data(driver, day.strftime("%Y-%m-%d"))


def get_sale_auction_data(start_date):
    sql_base = 'SELECT * FROM nifi_ethereum.transactions WHERE to_address = "0xb1690c08e213a35ed9bab7b318de14420fb57d8c" AND '
    date = datetime.strptime(start_date, "%Y-%m-%d")
    n = 365 * 6
    for i in range(n):
        day = date + timedelta(days=i)
        if day.strftime("%Y-%m-%d") == '2023-01-01':
            break
        sql = sql_base + 'day=\"' + day.strftime("%Y-%m-%d") + '\"'
        print(sql)
        pyperclip.copy(sql)

        get_data(driver, day.strftime("%Y-%m-%d"))


def get_log_data(start_date):
    sql_base = 'SELECT * FROM nifi_ethereum.logs WHERE address = "0x06012c8cf97bead5deae237070f9587f8e7a266d" AND '
    date = datetime.strptime(start_date, "%Y-%m-%d")
    n = 365 * 6
    for i in range(n):
        day = date + timedelta(days=i)
        if day.strftime("%Y-%m-%d") == '2023-01-01':
            break
        sql = sql_base + 'day=\"' + day.strftime("%Y-%m-%d") + '\"'
        print(sql)
        pyperclip.copy(sql)

        get_data(driver, day.strftime("%Y-%m-%d"))


def get_block_number():
    directory = '../raw data/core log'  # Replace with the actual directory path

    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter out directories from the list
    files = [file for file in files if os.path.isfile(os.path.join(directory, file))]

    # Get the newest file based on creation time
    newest_file = max(files, key=lambda file: os.path.getctime(os.path.join(directory, file)))

    df = pd.read_csv(os.path.join(directory, newest_file))
    block_numbers = df['logs_block_number'].unique()
    block_number = np.partition(block_numbers, -2)[-2]
    return block_number


def get_log_data_overlimit():
    sql_base = 'SELECT * FROM nifi_ethereum.logs WHERE address = "0x06012c8cf97bead5deae237070f9587f8e7a266d" AND '
    with open('dayoverlimit_log.txt') as f:
        dates = f.readlines()
    for date in dates:
        if date[-1] == '\n':
            date = date[:-1]
        day = datetime.strptime(date, '%Y-%m-%d')
        sql = sql_base + 'day=\"' + day.strftime('%Y-%m-%d') + '\"'
        sql += ' ORDER BY block_number ASC LIMIT 99999'
        print(sql)
        pyperclip.copy(sql)
        get_data(driver, day.strftime("%Y-%m-%d"))
        time.sleep(30)
        block_number = get_block_number()
        sql = sql_base + 'day=\"' + day.strftime('%Y-%m-%d') + '\"'
        sql += " AND block_number >\'" + str(block_number) + "\'"
        sql += ' ORDER BY block_number ASC'
        print(sql)
        pyperclip.copy(sql)
        get_data(driver, day.strftime("%Y-%m-%d"))


if __name__ == '__main__':
    driver = connect()

    # get_core_data(start_date='2022-01-12')
    # get_sale_auction_data(start_date='2020-07-21')
    get_log_data(start_date='2020-10-01')

    # get_log_data_overlimit()
