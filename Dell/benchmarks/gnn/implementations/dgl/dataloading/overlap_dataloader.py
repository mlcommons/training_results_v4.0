# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

import threading
import atexit
import torch
import os
from queue import Empty, Full, Queue


PYTHON_EXIT_STATUS = False

def _python_exit():
    global PYTHON_EXIT_STATUS
    PYTHON_EXIT_STATUS = True

atexit.register(_python_exit)

interleaver_timeout = int(os.environ.get("INTERLEAVER_TIMEOUT", "600"))


def _try_put_data(queue, data, done_event):
    while not done_event.is_set():
        try:
            queue.put(data, timeout=interleaver_timeout)
            break
        except Full:
            continue


def SamplingThread(
    device, dataloader_it,
    batch_queue, sample_start_queue,
    stream, done_event, sample_done_launch_event,
    repeat_input_after,
):
    torch.cuda.set_device(device)
    sample_iter_num = 0
    try:
        while not done_event.is_set():
            try:
                sample_flag = sample_start_queue.get(timeout=interleaver_timeout)
            except Empty:
                raise RuntimeError(
                    f"Time out when prefetcher wait for sample flag at {interleaver_timeout} seconds."
                )
            try:
                if sample_flag:
                    if repeat_input_after > 0 and sample_iter_num > repeat_input_after:
                        batch = held_batch
                    else:
                        with torch.cuda.stream(stream):
                            batch = next(dataloader_it)
                            if repeat_input_after > 0 and sample_iter_num == repeat_input_after:
                                held_batch = batch
                    sample_done_launch_event.set()
                else:
                    break

            except StopIteration:
                batch = None
                sample_done_launch_event.set()

            _try_put_data(batch_queue, (batch, None), done_event)
            sample_iter_num += 1

    except Exception as e:
        print(f"Catch exception in the SamplingThread: {e}")
        raise


def pre_feature_read(batch, feature_extractor, device, features):
    actual_batch = feature_extractor.extract_graph_structure(batch, device)
    batch_inputs, batch_labels = feature_extractor.extract_inputs_and_outputs(
        actual_batch, device, features
    )
    return (actual_batch, batch_inputs, batch_labels)


def EmbedThread(
    feature_extractor, device,
    features, embed_queue,
    embed_start_queue, stream,
    done_event, embed_done_launch_event
):
    torch.cuda.set_device(device)
    try:
        while not done_event.is_set():
            try:
                batch = embed_start_queue.get(timeout=interleaver_timeout)
            except Empty:
                raise RuntimeError(
                    f"Time out when prefetcher wait for the sampled batch at {interleaver_timeout} seconds."
                )

            if batch == None:
                actual_batch, batch_inputs, batch_labels = None, None, None
                exception = StopIteration
                embed_done_launch_event.set()
            else:
                with torch.cuda.stream(stream):
                    actual_batch, batch_inputs, batch_labels = pre_feature_read(
                        batch, feature_extractor, device, features
                    )
                    exception = None
                    embed_done_launch_event.set()

            _try_put_data(
                embed_queue,
                (actual_batch, batch_inputs, batch_labels, exception),
                done_event,
            )

    except Exception as e:
        print(f"Catch exception in the EmbedThread: {e}")
        raise


class PrefetchInterleaver:
    def __init__(
        self, dataloader, sample_stream, embed_stream, feature_extractor, device, features, repeat_input_after=-1,
    ):
        self.dataloader = dataloader
        self.sample_stream = sample_stream
        self.embed_stream = embed_stream
        self.feature_extractor = feature_extractor
        self.device = device
        self.features = features
        self.dataloader_len = len(self.dataloader)
        self.is_first_epoch = True
        self.cur_batch = None
        self.next_batch = None
        self.exception = None
        self.repeat_input_after = repeat_input_after
        self.shut_down_done = False

    def __iter__(self):
        self._status_init()
        self.dataloader_iter = iter(self.dataloader)
        sample_thread = threading.Thread(
            target=SamplingThread,
            args=(
                self.device,
                self.dataloader_iter,
                self.batch_queue,
                self.sample_start_queue,
                self.sample_stream,
                self._done_event,
                self._sample_done_launch_event,
                self.repeat_input_after,
            ),
            daemon=True,
        )
        embed_thread = threading.Thread(
            target=EmbedThread,
            args=(
                self.feature_extractor,
                self.device,
                self.features,
                self.embed_queue,
                self.embed_start_queue,
                self.embed_stream,
                self._done_event,
                self._embed_done_launch_event
            ),
            daemon=True,
        )
        sample_thread.start()
        embed_thread.start()
        self.sample_thread = sample_thread
        self.embed_thread = embed_thread

        self.pre_sample()
        self._sample_done_launch_event.wait(timeout=interleaver_timeout)
        self.embed_stream.wait_stream(self.sample_stream)
        try:
            self.next_batch, self.exception = self.batch_queue.get(
                timeout=interleaver_timeout
            )
        except Empty:
            raise RuntimeError(
                f"Time out when prefetcher gets next batch at {interleaver_timeout} seconds."
            )

        self.pre_sample()
        self.pre_embed_read()
        self.check_overlap_launch_end()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.sample_stream)
        torch.cuda.current_stream().wait_stream(self.embed_stream)
        self.sample_stream.wait_stream(torch.cuda.current_stream())
        self.embed_stream.wait_stream(torch.cuda.current_stream())

        try:
            (
                self.actual_batch,
                self.batch_inputs,
                self.batch_labels,
                self.exception,
            ) = self.embed_queue.get(timeout=interleaver_timeout)
        except Empty:
            raise RuntimeError(
                f"Time out when prefetcher gets embed data at {interleaver_timeout} seconds."
            )

        if self.actual_batch is None:
            raise self.exception
        else:
            self.cur_batch = self.next_batch

            try:
                self.next_batch, self.exception = self.batch_queue.get(
                    timeout=interleaver_timeout
                )
            except Empty:
                raise RuntimeError(
                    f"Time out when prefetcher gets next batch at {interleaver_timeout} seconds."
                )

        self.pre_sample()
        self.pre_embed_read()

        return self.cur_batch

    def _status_init(self):
        self.cur_batch = None
        self.next_batch = None
        self.actual_batch = None
        self.batch_inputs = None
        self.batch_labels = None
        self.embed_queue = Queue(1)
        self.batch_queue = Queue(1)
        self.sample_start_queue = Queue(1)
        self.embed_start_queue = Queue(1)
        self._done_event = threading.Event()
        self._sample_done_launch_event = threading.Event()
        self._embed_done_launch_event = threading.Event()
        self.shut_down_done = False

    def thread_cleanup(self):
        try:
            self.check_overlap_launch_end()

            self._done_event.set()
            self.sample_start_queue.put(0)
            self.embed_start_queue.put(None)

            self.sample_stream.synchronize()
            self.embed_stream.synchronize()

            self.sample_thread.join()
            self.embed_thread.join()
            self.shut_down_done = True
        except:
            pass

    def get_inputs_and_outputs(self):
        return (self.actual_batch, self.batch_inputs, self.batch_labels)

    def pre_sample(self):
        self._sample_done_launch_event.clear()
        try:
            self.sample_start_queue.put(1)
        except Full:
            raise RuntimeError("Time out when prefetcher set sample_start_queue.")

    def pre_embed_read(self):
        self._embed_done_launch_event.clear()
        try:
            self.embed_start_queue.put(self.next_batch)
        except Full:
            raise RuntimeError("Time out when prefetcher set embed_start_queue.")

    def __len__(self):
        return self.dataloader_len

    def check_overlap_launch_end(self):
        self._sample_done_launch_event.wait(timeout=interleaver_timeout)
        self._embed_done_launch_event.wait(timeout=interleaver_timeout)

    def __del__(self):
        if PYTHON_EXIT_STATUS is True or PYTHON_EXIT_STATUS is None:
            return
        if not self.shut_down_done:
            self.thread_cleanup()
