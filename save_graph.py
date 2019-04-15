#!/usr/bin/env python3

import tensorflow as tf
from prnet.predictor import PosPrediction

PRN_PATH = './data/net-data/256_256_resfcn256_weight'
GRAPH_PATH = PRN_PATH + '.meta'

RESOLUTION_IN = 256
RESOLUTION_OUT = 256

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    pos_predictor = PosPrediction(RESOLUTION_IN, RESOLUTION_OUT)

    tf.logging.info("Restoring graph...")
    pos_predictor.restore(PRN_PATH)

    tf.logging.info("Saving graph...")
    pos_predictor.saver.export_meta_graph(GRAPH_PATH)
