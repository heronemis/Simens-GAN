import os
import scipy.misc
import numpy as np
import time
from datetime import datetime
from glob import glob

from model import DCGAN
from utils import pp, visualize, to_json

import newUtils

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 40, "Epoch to train [25]")
flags.DEFINE_float("learning_rate_D", 0.0002, "Learning rate of Discriminator for adam [0.0002]") #00006
flags.DEFINE_float("learning_rate_G", 0.0002, "Learning rate of Generator for adam [0.0002]")
flags.DEFINE_float("beta1_D", 0.5, "Momentum term of Discriminator for adam [0.5]")
flags.DEFINE_float("beta1_G", 0.5, "Momentum term of Generator for adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")

flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpeg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

flags.DEFINE_integer("eval_size", 10, "Selcted number times the batch size select the number of images used when samples are evaluated")


flags.DEFINE_boolean("shuffle_data", False, "Shuffle training data before training [False]")
flags.DEFINE_boolean("improved_z_noise", False, "Use Z noise based on training images [False]")
flags.DEFINE_boolean("static_z", False, "Use the Z noise during each epoch of training[False]")
flags.DEFINE_boolean("minibatch_discrimination", False, "Use of Minibatch Discrimination [False]")
flags.DEFINE_integer("tournament_selection", 1, "0 is turned off. 1 will select the best images from a large selection while 2 will select the worst images. [0,1,2,3]")


FLAGS = flags.FLAGS


def main(_):



    pp.pprint(flags.FLAGS.__flags)
    # pp.pprint(flags.FLAGS.__parse)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if (FLAGS.dataset == "cat"):
        data = glob(os.path.join("./data", "cat/*", "*.jpg"))
    else:
        data = glob(os.path.join("./data", FLAGS.dataset, "*.jpeg"))
    if(len(data) == 0):
        print("Did not find any photos with extension .jpeg")

        data = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
        print("Trying extension .jpg - Found",len(data),"images :)")
        FLAGS.input_fname_pattern = "*.jpg"


    newUtils.createFolderName(FLAGS)
    # print(FLAGS.sample_dir)

    newUtils.createConfingCSV(FLAGS.sample_dir,FLAGS)
    # return

    pp.pprint(flags.FLAGS.__flags)

    # print(FLAGS.sample_dir.default_value)

    # if not os.path.exists(path):
    #     print("path", path, "not found. Creating new folder")
    #     os.makedirs(path)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                y_dim=10,
                c_dim=1,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)
        else:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                c_dim=FLAGS.c_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)

        if FLAGS.is_train:



            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")


                # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
                #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
                #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
                #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
                #                 [dcgan.h4_w, dcgan.h4_b, None])

                # Below is codes for visualization
                # OPTION = 1
                # visualize(sess, dcgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()
