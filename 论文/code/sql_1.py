import time
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import win32con
import win32api
import pyperclip

key_map = {"V": 86, 'CONTROL': 17}


def press_key(key_code):
    win32api.keybd_event(key_code, win32api.MapVirtualKey(key_code, 0), 0, 0)


def release_key(key_code):
    win32api.keybd_event(key_code, win32api.MapVirtualKey(key_code, 0), win32con.KEYEVENTF_KEYUP, 0)


def connect():
    option = webdriver.ChromeOptions()
    option.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
    chromedriver_path = r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe'
    driver = webdriver.Chrome(options=option, executable_path=chromedriver_path)
    return driver


def get_data(driver, date):
    driver.find_element(By.XPATH,
        '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[1]/div[1]/button[1]'
                        ).click()
    driver.find_element(By.XPATH,
        '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[2]/div[6]/div[1]'
        '/div/div/div/div[5]/div/pre').click()
    press_key(key_map["CONTROL"])
    press_key(key_map["V"])
    release_key(key_map["CONTROL"])
    release_key(key_map["V"])
    time.sleep(2)

    driver.find_element(By.XPATH,
        '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[3]/button[1]'
                        ).click()

    while True:
        try:
            driver.find_element(By.XPATH,
                '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[1]/div/div[3]/button[2]'
                                ).click()
            time.sleep(2)
            driver.find_element(By.XPATH,
                '/html/body/div[@class="el-dialog__wrapper queryDialogNewQuery"]/div/div[2]/div/form/div[1]/div[1]/div/div/input'
                                ).send_keys(date)
            time.sleep(2)
            break
        except:
            time.sleep(2)
    # 保存
    driver.find_element(By.XPATH,
        '/html/body/div[@class="el-dialog__wrapper queryDialogNewQuery"]/div/div[2]/div/form/div[2]/button'
                        ).click()
    time.sleep(4)
    # 下载
    try:
        driver.find_element(By.XPATH,
        '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[2]/div/div[2]/button[2]'
                            ).click()
    except:
        time.sleep(4)
        driver.find_element(By.XPATH,
            '//*[@id="hive-query-container"]/div[2]/div/div[1]/div[2]/div/div[2]/button[2]'
                            ).click()

    time.sleep(2)


if __name__ == '__main__':
    driver = connect()

    sql_base = 'SELECT * FROM nifi_ethereum.transactions WHERE to_address = "0x06012c8cf97bead5deae237070f9587f8e7a266d" AND '

    start_date = '2017-11-23'
    date = datetime.strptime(start_date, "%Y-%m-%d")
    n = 365*3
    for i in range(n):
        day = date + timedelta(days=i)
        if day.strftime("%Y-%m-%d") == '2023-01-01':
            break
        sql = sql_base + 'day=\"' + day.strftime("%Y-%m-%d") + '\"'
        print(sql)
        pyperclip.copy(sql)

        get_data(driver, day.strftime("%Y-%m-%d"))