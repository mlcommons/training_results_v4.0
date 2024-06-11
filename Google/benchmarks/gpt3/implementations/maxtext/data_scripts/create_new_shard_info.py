"""Script to generate right dataset_info.json."""
# pylint: disable=redefined-outer-name, logging-fstring-interpolation
import argparse
import json
import logging
import os
import time
import tensorflow.compat.v1 as tf
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Script to generate right dataset_info.json')

parser.add_argument(
    '--gcs-prefix',
    type=str,
    default='',
    help='Prefix of the output TFRecords, including path of directory.')

parser.add_argument(
    '--template-json',
    type=str,
    default='gs://mlperf-llm-public2/c4/en/3.0.4/dataset_info.json',
    help='template json file')
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)


def get_num_examples(tfrecord_path: str) -> int:
  return len(list(tf.data.TFRecordDataset(tfrecord_path)))


def get_index_num_examples(tfrecord_path: str) -> tuple[int, int]:
  base_filename = os.path.basename(tfrecord_path)
  # example: c4-train.tfrecord-01006-of-01024
  index = int(base_filename.split('-')[2].strip())
  num_examples = get_num_examples(tfrecord_path)
  return index, num_examples

if __name__ == '__main__':
  tic = time.time()
  tf.enable_eager_execution()

  gcs_prefix = args.gcs_prefix
  template_json = args.template_json
  output_dataset_json_path = os.path.join(gcs_prefix, 'dataset_info.json')
  tfrecord_paths = tf.io.gfile.glob(os.path.join(gcs_prefix, '*tfrecord*'))

  with tf.io.gfile.GFile(output_dataset_json_path, mode='w') as fout:
    with tf.io.gfile.GFile(template_json, mode='r') as fin:
      dataset_info = json.load(fin)
      dataset_info['splits'][0]['numShards'] = len(tfrecord_paths)
      shard_lengths_list = [0] * len(tfrecord_paths)
      logging.info('gather num_examples for each tfrecord file')
      for tfrecord_path in tqdm(tfrecord_paths):
        index, num_examples = get_index_num_examples(tfrecord_path)
        shard_lengths_list[index] = num_examples
        logging.info(f'{tfrecord_path=}: {num_examples=}')

      logging.info(f'{sum(shard_lengths_list)=}')
      dataset_info['splits'][0]['shardLengths'] = shard_lengths_list

    fout.write(json.dumps(dataset_info))
