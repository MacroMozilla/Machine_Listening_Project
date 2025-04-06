import functools
from datetime import datetime

import pandas as pd


class Watch:
    def __init__(self, fctn=pd.Timestamp.now):
        self.fctn = fctn
        self.cur_timestamp = self.fctn()
        self.records = []
        self.see_timedelta()

    def see_timedelta(self):
        now_timestamp = self.fctn()
        res = now_timestamp - self.cur_timestamp
        self.records.append(res)
        self.cur_timestamp = now_timestamp
        return res

    def see_seconds(self):
        return round(self.see_timedelta().total_seconds(), 6)

    def total_timedelta(self):
        totalTime = sum(self.records, pd.Timedelta(seconds=0))
        return totalTime

    def total_seconds(self):
        return round(self.total_timedelta().total_seconds(), 6)

    def check_timedelta(self):
        now_timestamp = self.fctn()
        res = now_timestamp - self.cur_timestamp
        return res

    def check_seconds(self):
        return round(self.check_timedelta().total_seconds(), 6)


def watch_time(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        w = Watch()
        res = func(*args, **kwargs)

        print(f"Time Cost : {func.__name__} {w.see_seconds()}")
        return res

    return wrap


def datetime2sent(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')


def sent2datetime(sent):
    return datetime.strptime(sent, '%Y-%m-%d %H:%M:%S.%f')


def get_current_time():
    """获取当前时间戳"""
    return datetime.now()

if __name__ == '__main__':
    pass
    w = Watch()
