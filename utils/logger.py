import logging
import os
import time

class Log(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self):
        cur_path = os.path.dirname(os.path.realpath(__file__))  # log_path是存放日志的路径
        log_path = os.path.join(os.path.dirname(cur_path), 'logs')
        if not os.path.exists(log_path):
            os.mkdir(log_path)  # 如果不存在这个logs文件夹，就自动创建一个
        self.logger = logging
        self.logger = logging.getLogger(os.path.join(log_path, 'all-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))))
        format_str = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        console_formatter = logging.Formatter(
         fmt='[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] [%(levelname)s]- %(message)s',
         datefmt='%Y-%m-%d  %H:%M:%S',
        )
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        sh.setFormatter(console_formatter)
        th = logging.FileHandler(filename=os.path.join(log_path, 'all-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))), mode='w')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

Logger = Log() 