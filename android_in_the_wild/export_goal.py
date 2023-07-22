# %% [markdown]
# Copyright 2023 The Google Research Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import datetime
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
            logging.FileHandler(f"export-goal-android-in-wild-{'{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())}.log"),
        ]
)

logger = logging.getLogger(__name__)

def global_exception_handler(type, value, error_traceback):
    """
    refer to https://stackoverflow.com/questions/7075200/converting-exception-to-a-string-in-python-3
    """
    logger.exception(f"Uncaught exception {str(value)}")
    logger.error(str(type))
    logger.error(f"\n\t{''.join(traceback.format_tb(error_traceback))}")
    sys.exit()

sys.excepthook = global_exception_handler

import sys
sys.path.append('.')
sys.path.append('..')
import json

# %%
from android_in_the_wild import visualization_utils
import tensorflow as tf
import random

# %%
dataset_name = 'google_apps'  #@param ["general", "google_apps", "install", "single", "web_shopping"]

dataset_directories = {
    'general': 'gs://gresearch/android-in-the-wild/general/*',
    'google_apps': 'gs://gresearch/android-in-the-wild/google_apps/*',
    'install': 'gs://gresearch/android-in-the-wild/install/*',
    'single': 'gs://gresearch/android-in-the-wild/single/*',
    'web_shopping': 'gs://gresearch/android-in-the-wild/web_shopping/*',
}

# %%
filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()

# %%
def get_episode(dataset):
  """Grabs the first complete episode."""
  episode = []
  episode_id = None
  for d in dataset:
    ex = tf.train.Example()
    ex.ParseFromString(d)
    ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
    goal = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
    if episode_id is None:
      episode_id = ep_id
      episode.append(ex)
    elif ep_id == episode_id:
      episode.append(ex)
    else:
      break
  return episode

def get_all_episode(dataset):
  episodes = {}
  episode = []
  episode_id = None
  for d in dataset:
    ex = tf.train.Example()
    ex.ParseFromString(d)
    ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
    if episode_id is None:
      episode_id = ep_id
      episode.append(ex)
    elif ep_id == episode_id:
      episode.append(ex)
    else:
      # ep_id != episode_id
      episodes[ep_id] = episode; episode = []
      episode_id = ep_id
      episode.append(ex)
  return episode

def get_all_episodes_num(dataset):
  episodes = {}
  episode = []
  episode_id = None
  cnt = 0
  for d in dataset:
    ex = tf.train.Example()
    ex.ParseFromString(d)
    ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
    if episode_id is None:
      episode_id = ep_id; cnt += 1
    elif ep_id == episode_id:
      pass
    else:
      # ep_id != episode_id
      # episodes[ep_id] = episode; episode = []
      episode_id = ep_id; cnt += 1
      # episode.append(ex)
  return cnt

# %%
# ep = get_episode(raw_dataset)
# visualization_utils.plot_episode(ep, show_annotations=True, show_actions=True)

# %%
# eps = get_all_episode(raw_dataset)
# print(len(eps))

# %%
# eps_num = get_all_episodes_num(raw_dataset)

# %%


def export_goal_for_datasets(dataset, dataset_name='google_apps'):
  episode_id_to_goal = {}
  episode_id = None
  for i, d in enumerate(dataset):
    ex = tf.train.Example()
    ex.ParseFromString(d)
    ep_id = ex.features.feature['episode_id'].bytes_list.value[0].decode('utf-8')
    goal = ex.features.feature['goal_info'].bytes_list.value[0].decode('utf-8')
    if episode_id is None:
      episode_id = ep_id
    elif ep_id == episode_id:
      pass
    else:
      # ep_id != episode_id
      episode_id_to_goal[ep_id] = goal
      episode_id = ep_id
    if i > 0 and i % 10000 == 0:
      with open(f'{dataset_name}.txt', 'w') as f:
        json.dump(episode_id_to_goal, f)

# see: google_apps has been done
for dataset_name in  ["general", "install", "single", "web_shopping"]:
  start = datetime.datetime.now()
  filenames = tf.io.gfile.glob(dataset_directories[dataset_name])
  raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').as_numpy_iterator()
  export_goal_for_datasets(raw_dataset, dataset_name)
  logger.info(f'Export goal for {dataset_name} done in {(datetime.datetime.now() - start) / 3600.} hours')