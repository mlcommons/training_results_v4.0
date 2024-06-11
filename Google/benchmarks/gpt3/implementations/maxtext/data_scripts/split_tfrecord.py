"""Script to split a single TFRecord file into multiple ones."""
# pylint: disable=redefined-outer-name, line-too-long, logging-fstring-interpolation

import argparse
import logging
import os
import time
import tensorflow.compat.v1 as tf

parser = argparse.ArgumentParser(
    description="Split a single TFRecord file into multiple ones.")
parser.add_argument(
    "--input_tfrecord",
    type=str,
    default="",
    help="Path to the input TFRecord file.")
parser.add_argument(
    "--num_splits",
    type=int,
    default=4,
    help="Number of TFRecords to split the input into.")
parser.add_argument(
    "--output_prefix",
    type=str,
    default="",
    help="Prefix of the output TFRecords, including path of directory.")
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def get_index(input_filename: str) -> int:
  # example: tfrecord-01006-of-01024
  base_filename = os.path.basename(input_filename)
  index = int(base_filename.split("-")[2].strip())
  return index


def get_total_count(input_filename: str) -> int:
  # example: tfrecord-01006-of-01024
  base_filename = os.path.basename(input_filename)
  total_count = int(base_filename.split("-")[-1].strip())
  return total_count

if __name__ == "__main__":
  tic = time.time()
  tf.enable_eager_execution()

  input_filename = args.input_tfrecord

  input_index = get_index(input_filename)
  input_total_count = get_total_count(input_filename)

  num_splits = args.num_splits
  output_filenames = [args.output_prefix + f".tfrecord-{input_index * num_splits + i:05}-of-{input_total_count * num_splits:05}"
                      for i in range(num_splits)]
  logging.info(output_filenames)
  writers = [tf.io.TFRecordWriter(output_filename)
             for output_filename in output_filenames]
  num_output_examples = [0 for _ in range(num_splits)]
  d = tf.data.TFRecordDataset(input_filename)
  input_examples = list(d)
  num_examples = len(input_examples)
  split_num = 0
  for i in range(num_examples):
    if 1.0 * (split_num + 1) * num_examples / num_splits + 1e-6 <= i:
      split_num += 1
    writers[split_num].write(input_examples[i].numpy())
    num_output_examples[split_num] += 1

  for j in range(num_splits):
    logging.info("  %s: %d examples",
                 output_filenames[j], num_output_examples[j])
    writers[j].close()

  toc = time.time()
  logging.info("Split %d examples into %d files in %.2f sec",
               num_examples, num_splits, toc - tic)
