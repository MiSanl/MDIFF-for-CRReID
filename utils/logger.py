import logging
import os
import sys
import os.path as osp
import time


def setup_logger(name, save_dir, if_train):
    logger = logging.getLogger(name)  # 创建并返回一个名为name的Logger对象，用于记录日志信息
    logger.setLevel(logging.DEBUG)
    # 创建一个输出到控制台的StreamHandler对象，用于将日志信息打印到终端上
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # 创建一个Formatter对象，用于设置日志信息输出的格式;
    # 其中，%(asctime)s表示日志记录时间，%(name)s表示Logger对象名称，%(levelname)s表示日志级别，%(message)s表示日志消息;
    # datefmt = '%Y-%m-%d %H:%M:%S'表示将日志记录时间以指定格式进行输出
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s",datefmt  = '%Y-%m-%d %H:%M:%S')
    # 将Formatter对象应用到StreamHandler对象中，设置日志输出格式
    ch.setFormatter(formatter)
    # 将StreamHandler对象添加到Logger对象中，表示将日志信息输出到终端
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            # 创建一个输出到文件的FileHandler对象，用于将日志信息写入文件
            # fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            fh = logging.FileHandler(os.path.join(save_dir, 'train_log_{}.txt'.format(time_str)), mode='w')
        else:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            fh = logging.FileHandler(os.path.join(save_dir, "test_log_{}.txt".format(time_str)), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
