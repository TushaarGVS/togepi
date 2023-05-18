
import random
import logging

import tensorflow as tf

from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

dataset.save_to_disk("/mnt/beegfs/bulk/stripe/lm865/WikiData/")

#dataset.save_to_disk("/Users/lucasmolter/Documents/Cornell/Courses/Spring2023/CS5223 /togepi/Lucas/datatest")

