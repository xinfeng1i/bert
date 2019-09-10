# -*- coding: utf-8 -*-

from __future__ import print_function
import json
import os
import struct
import sys
import platform
import re
import time
import traceback
import requests


def __get_ip(nic='eth1'):
    '''
    Get ipaddr on your machine.
    :param nic: NIC(Network interface card) name.
    :return: a string, ip address on the NIC.
    '''
    if platform.system() == 'Windows':
        return '127.0.0.1'
    else:
        try:
            my_addr = os.popen(
                "ifconfig | grep -A 1 %s|tail -1| awk '{print $2}'" % nic).read()
            ip = re.search(r'(?<![\.\d])(?:25[0-5]\.|2[0-4]\d\.|[01]?\d\d?\.)'
                           r'{3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?![\.\d])', my_addr).group()
        except AttributeError:
            ip = 'localhost'
        return ip
		

def get_ip():
    ip = __get_ip(nic='eth1')
    if ip == "localhost":
        ip = __get_ip(nic='eth0')
    return ip
	

report_ip = os.environ.get('CHIEF_IP', '')
if not report_ip:
    report_ip = get_ip()
	

def report_progress(progress):
    """
    向worker上报训练进度
    :param progress: dict, https://docs.qq.com/doc/DWUREanR2bnFCbklh?opendocxfrom=admin里约定的通信协议
    :return:
    """
    url = "http://%s:%s/v1/worker/report-progress" % (report_ip, 8080)
    try:
        response = requests.post(url, json=json.dumps(progress))
    except Exception as e:
        print("send progress info to worker failed!\nprogress_info: %s, \n%s" % (progress, traceback.format_exc()))
        return False, str(e)
    if response.status_code != 200:
        print("send progress info to worker failed!\nprogress_info: %s, \nreason: %s" % (progress, response.reason))
        return False, response.text
    return True, ""


def report_error(code, msg=""):
    """
    向worker上报错误信息
    :param code: 错误码
    :param msg: 出错原因
    :return:
    """
    progress = {"type": "error", "code": code, "msg": msg}
    return report_progress(progress)


def heartbeat():
    """
    通知worker，trainer还在正常运行
    :return:
    """
    progress = {"type": "alive"}
    return report_progress(progress)


def job_completed():
    """
    通知worker，训练任务结束
    :return:
    """
    progress = {"type": "completed"}
    return report_progress(progress)


if __name__ == "__main__":
  ## 不同上报的测试样例
    ##1 测试上报心跳
    for i in range(3):
        heartbeat()
    ## 2 上报训练进度
    progress = {"step": 0, "type": "train", "loss": 1.5}
    for i in range(3):
        time.sleep(3*60)
        ret, msg = report_progress(progress)
        print("ret: %s, %s" % (ret, msg))
    
    ## 3 上报测试进度
    progress = {"step": 0, "type": "test", "accracy": 0.3}
    for i in range(3):
        time.sleep(3*60)
        ret, msg = report_progress(progress)
        print("ret: %s, %s" % (ret, msg))
    
    ## 4 上报结束信息
    job_completed()
