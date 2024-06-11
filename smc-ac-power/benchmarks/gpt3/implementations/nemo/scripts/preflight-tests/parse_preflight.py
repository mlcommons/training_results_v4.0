#!/bin/python3

import os
import re
import sys
import json
import csv

def extract_time(line):
    time = re.search(r'\d+', line.split("0: :::")[1])
    return int(time.group()) / 1000


def get_time(filename):
    result = []
    with open(filename, "r") as f:
        content = f.readlines()
    content = [l[14:] for l in content if "train_step_timing" in l]
    for i in content:
        try:
            a = json.loads(i)
        except:
            a = json.loads(i[2:])
        f = float(a['value']['train_step_timing in s'])
        result.append(f)
    if len(result) == 0:
        return 0, 0, False

    first_step = result[0]
    result = result[1:]
    return first_step, sum(result)/len(result), True

# get time from timestamps as there is a bug in the time reported by train_step_timing
# Todo: fix bug in train_step_timing reporting
def get_time_timestamp(filename):
    train_valid_timestamps = []
    is_train = []

    # get timestamps from all train and eval steps and from run_start tag
    # also keep track of whether each timestamp is train/eval step
    # These regexes match lines in form:
    #0: :::MLLOG {"namespace": "", "time_ms": 1713455596653, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/workspace/llm/custom_callbacks.py", "lineno": 217, "first_epoch_num": 524288, "epoch_count": 786432}}
    #0: Step 19 train_step_timing 1.9324331283569336 timestamp: 1713522620986.0247
    for line in open(filename, "r"):
        if "run_start" in line:
            current_time = json.loads(line.split(":::MLLOG ")[1])["time_ms"]
            train_valid_timestamps.append(current_time)
        if match1 := re.search(r"(train|validation)_step_timing", line):
            is_train.append(match1.group(1) == 'train')
            match2 = re.search(r"(?:\"time_ms\"|timestamp): (\d+)", line)
            current_time = float(match2.group(1))
            train_valid_timestamps.append(current_time)

    if len(train_valid_timestamps) < 2:
        return 0, 0, 0, False

    # we can get step times by subtracting previous timestamp from 
    train_valid_times = [(cur - prev)/1000 for prev, cur in zip(train_valid_timestamps, train_valid_timestamps[1:])]

    # drop all eval times and keep only train times
    train_indices = [idx for idx,is_train_val in enumerate(is_train) if is_train_val == 1]
    train_times = [train_valid_times[idx] for idx in range(len(train_valid_times)) if idx in train_indices]

    first_step = train_times[0]
    train_times = train_times[1:]
    return first_step, sum(train_times)/len(train_times), sorted(train_times)[len(train_times) // 2], True


def llm_nccl_test(lines):
    if len(lines) < 1:
        return "none", "none", "none"
    # each line contains -
    # DP, TP, PP, message size(B), count(elements), data type, redop, algo BW(GB/s), bus BW(GB/s)
    zipped_lines = list(zip(*lines))
    algobw = [float(x) for x in zipped_lines[7]]
    busbw = [float(x) for x in zipped_lines[8]]
    return min(algobw), sum(algobw)/len(algobw), sum(busbw)/len(busbw)

#:::DLPAL /mnt/resource_nvme/mlperf/nvidian+swdl+spalsamudram+optimized_9810757_nccl_fix.sqsh 16613 1 phx11c4-876 azure DGXH100_1x8x32x8x1
def main(argv):
    with open(argv[0], "r") as f:
        hostnames = [h.strip() for h in f.readlines()]
    # print(hostnames)

    logfiles = [os.path.join(argv[1], f) for f in os.listdir(argv[1]) if ".out" in f and "slurm" in f]
    # print(logfiles)

    not_started = []
    not_finished = []
    results = []
    for logfile in logfiles:
        with open(logfile, "r") as f:
            lines = f.readlines()
        content = "\n".join(lines)
        if "DLPAL" not in content:
            node = [l for l in lines if "+ MLPERF_SLURM_FIRSTNODE" in l][0].split("=")[-1].strip()
            try:
                reason = [l for l in lines if "ERROR" in l][0].strip()
            except:
                reason = ""
            not_started.append((logfile, node, reason))
            try:
                hostnames.remove(node)
            except ValueError:
                pass
            continue

        dlpal = [l for l in lines if l.startswith(":::DLPAL")][0].split(" ")
        container = dlpal[1].split("/")[-1]
        jobid = dlpal[2]
        nnodes = dlpal[3]
        node = dlpal[4]
        config = dlpal[6].strip()
        if "run_stop" not in content:
            not_finished.append((logfile, node))
            try:
                hostnames.remove(node)
            except ValueError:
                pass
            continue
        try:
            bus_bw = [l for l in lines if "Avg bus bandwidth" in l][0].split(":")[-1].strip()
        except:
            bus_bw = "none"
        message_lines = [l.split() for l in lines if "       104857600" in l]
        min_algobw, avg_algobw, avg_busbw = llm_nccl_test(message_lines)
        first_step, avg_step, median_step, status = get_time_timestamp(logfile)
        if not status:
            not_finished.append((logfile, node))
        else:
            results.append((node, first_step, avg_step, median_step, bus_bw, nnodes, jobid, config, container, min_algobw, avg_algobw, avg_busbw))
        try:
            hostnames.remove(node)
        except:
            pass

    # Process successful results
    # sort based on AvgStep column
    results.sort(key=lambda x: float(x[2]))
    AvgStep = [x[2] for x in results]
    stats_data = [min(AvgStep), max(AvgStep), sum(AvgStep)/len(AvgStep), AvgStep[len(AvgStep) // 2]]
    # compute std as sqrt of variance using just standard libraries
    stats_data.append(sum(pow(x-stats_data[2], 2) for x in AvgStep) / len(AvgStep)**0.5)
    # round stats data to 2 decimal places
    stats_data = [round(x, 2) for x in stats_data]
    # round results to 3 decimal spaces
    for idx, r in enumerate(results):
        results[idx] = [round(val,3) if isinstance(val, float) else val for val in r]

    # Output results
    print("Missing")
    print("\n".join(hostnames))
    print("Not started properly")
    for c in not_started:
        print(*c)
    print("Not finished properly")
    for c in not_finished:
        print(*c)
    print(f"scontrol show hostname {','.join([n for _, n in not_finished])} > hostfile.crashed")
    print("Successful")
    stats_columns = ["MinAvgStep", "MaxAvgStep", "AvgAvgStep", "MedianAvgStep", "StdAvgStep"]
    print(",".join(stats_columns))
    print(*stats_data, sep=",")
    success_columns = ["NodeID", "FirstStep", "AvgStep", "MedianStep", "BusBW", "NNodes", "JobID", "Config", "Container", "MinAlgoBW", "AvgAlgoBW", "AvgBusBW"]
    print("\n")
    print(",".join(success_columns))
    for r in results:
        print(*r, sep=",")

    # write successful results to success.csv
    try:
        with open("success.csv", "w") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(stats_columns)
            writer.writerow(stats_data)
            writer.writerow("")
            writer.writerow(success_columns)
            for r in results:
                writer.writerow(r)
    except:
        print("Unable to create success.csv in current directory, please look at STDOUT for results")


if __name__ == '__main__':
    main(sys.argv[1:])
