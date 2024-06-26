# Copyright (c) 2021-2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from time import time
import mxnet as mx
from mxnet import cuda_utils as cu
from mlperf_log_utils import mllogger
from collections import defaultdict

class Metricstats(object):
    def __init__(self):
        self.total = 0
        self.count = 0
        self.min = 1000000000
        self.max = 0
    def addtag(self, dur):
        self.total += dur
        self.count += 1
        if dur < self.min:
           self.min = dur
        if dur > self.max:
           self.max = dur
    def getstats(self):
        return self.total, self.count, self.min, self.max
    def getcount(self):
        return self.count

class ScaleoutBridge(object):
    FWD_TIME = 'fwd_time'
    BWD_TIME = 'bwd_time'
    OPT_TIME = 'opt_time'
    LOAD_TIME = 'load_time'
    EVAL_TIME = 'eval_time'
    ITER_TIME = 'iter_time'
    EPOCH_TIME = 'epoch_time'

    def __init__(self, qmax, time_tags, nvtx_flag, deviceid):
        print("Scaleout performance bridge is running ...")
        self.qmax = qmax
        self.time_tags = time_tags
        self.nvtx_flag = nvtx_flag
        self.deviceid = deviceid
        self.bridgestats = defaultdict(Metricstats)
        self.start_epoch = 0
        self.start_eval = 0
        self.start_iter = 0
        '''tracking one tag at a time'''
        self.start_time = 0

    def push_nvtx(self, tag):
        cu.nvtx_range_push(tag)

    def pop_nvtx(self):
        cu.nvtx_range_pop()

    def print_tag(self, tag, dur):
        mllogger.event(key=tag, val={'r':self.deviceid, 't':dur}, uniq=False)

    def add_tag(self, tag, dur):
        self.bridgestats[tag].addtag(dur)
        if tag == self.ITER_TIME:
            if self.bridgestats[tag].getcount() > self.qmax:
                self.printstats()
                return 0
        return 1

    def start_prof(self, tag):
        if self.time_tags:
            mx.nd.waitall()
            if tag == self.ITER_TIME:
                self.start_iter = time()
            else:
                self.start_time = time()
        if self.nvtx_flag:
            self.push_nvtx(tag)

    def stop_prof(self, tag):
        if self.time_tags:
            mx.nd.waitall()
            if tag == self.ITER_TIME:
                if not self.add_tag(tag, time()-self.start_iter):
                    self.printstats()
                    self.time_tags = 0
                self.start_iter = 0
            else:
                self.add_tag(tag, time()-self.start_time)
                self.start_time = 0

        if self.nvtx_flag:
            self.pop_nvtx()
        return self.time_tags

    def stop_start_prof(self, tag1, tag2):
        if self.time_tags:
            mx.nd.waitall()
            new_start_time = time()
            if not self.add_tag(tag1, new_start_time-self.start_time):
                self.printstats()
                self.time_tags = 0
            self.start_time = new_start_time
        if self.nvtx_flag:
            self.pop_nvtx()
            self.push_nvtx(tag2)

    def start_epoch_prof(self):
        mx.nd.waitall()
        self.start_epoch = time()
        cu.cuda_profiler_start()

    def stop_epoch_prof(self):
        self.printstats()
        mx.nd.waitall()
        cu.cuda_profiler_stop()
        self.print_tag(self.EPOCH_TIME, time()-self.start_epoch)

    def start_eval_prof(self):
        mx.nd.waitall()
        self.start_eval = time()

    def stop_eval_prof(self):
        self.printstats()
        mx.nd.waitall()
        self.print_tag(self.EVAL_TIME, time()-self.start_eval)

    def printstats(self):
        if not self.time_tags:
            return
        for tag in self.bridgestats:
            self.printstat(tag)
        self.bridgestats.clear()

    def printstat(self, tag):
        total, count, minimum, maximum = self.bridgestats[tag].getstats()
        mllogger.event(key=tag+'_total', val={'r':self.deviceid, 't':total}, uniq=False)
        mllogger.event(key=tag+'_count', val={'r':self.deviceid, 't':count}, uniq=False)
        mllogger.event(key=tag+'_min', val={'r':self.deviceid, 't':minimum}, uniq=False)
        mllogger.event(key=tag+'_max', val={'r':self.deviceid, 't':maximum}, uniq=False)
class EmptyObject(object):
    def start_prof(self, tag):
        pass
    def stop_prof(self, tag):
        return 1
    def stop_start_prof(self, tag1, tag2):
        return 1
    def start_epoch_prof(self):
        pass
    def stop_epoch_prof(self):
        return 1
    def start_eval_prof(self):
        pass
    def stop_eval_prof(self):
        return 1

class ScaleoutBridge_Epoch(object):
    def __init__(self, deviceid):
        print("Scaleout performance bridge-epoch only is running ...")
        self.start_time = 0
        self.deviceid = deviceid
    def start_prof(self, tag):
        pass
    def stop_prof(self, tag):
        pass
    def stop_start_prof(self, tag1, tag2):
        pass
    def start_epoch_prof(self):
        mx.nd.waitall()
        self.start_time = time()
    def stop_epoch_prof(self):
        mx.nd.waitall()
        mllogger.event(key='epoch_time', val={'r':self.deviceid, 't':time()-self.start_time}, uniq=False)
    def start_eval_prof(self):
        pass
    def stop_eval_prof(self):
        return 1

def init_bridge(deviceid):
    time_tags = int(os.getenv('TIME_TAGS', 0))
    nvtx_flag = int(os.getenv('NVTX_FLAG', 0))
    epoch_only = int(os.getenv('EPOCH_PROF', 0))
    sbridge = EmptyObject()
    if time_tags or nvtx_flag:
        sbridge = ScaleoutBridge(1000, time_tags, nvtx_flag, deviceid)
    elif epoch_only:
        sbridge = ScaleoutBridge_Epoch(deviceid)

    return sbridge
