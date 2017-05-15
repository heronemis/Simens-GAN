import os
import scipy.misc
import numpy as np
import time
from glob import glob
from datetime import datetime

from model import DCGAN
from utils import pp, visualize, to_json


from ops import *
from utils import *
import newUtils

import tensorflow as tf

#  sudo python3 main.py --dataset celebA --input_height=108 --is_train --is_crop True
#  sudo python3 gamCompare.py --dataset celebA --input_height=108  --is_crop True

flags = tf.app.flags
flags2 = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
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




flags.DEFINE_boolean("shuffle_data", False, "Shuffle training data before training [False]")
flags.DEFINE_boolean("improved_z_noise", False, "Use Z noise based on training images [False]")
flags.DEFINE_boolean("static_z", False, "Use the Z noise during each epoch of training[False]")
flags.DEFINE_boolean("minibatch_discrimination", False, "Use of Minibatch Discrimination [False]")
flags.DEFINE_integer("tournament_selection", 0, "0 is turned off. 1 will select the best images from a large selection while 2 will select the worst images. [0,1,2]")

flags.DEFINE_string("gan1", "errorMissingGan1Checkpoint", "Checkpoint folder for GAN 1")
flags.DEFINE_string("gan2", "errorMissingGan2Checkpoint", "Checkpoint folder for GAN 2")


FLAGS = flags.FLAGS





def main(_):






    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    testDatasetSize = 120*FLAGS.batch_size
    sampleDatasetSize = 800 #*FLAGS.batch_size
    data = glob(os.path.join("./data", FLAGS.dataset, "*.jpeg"))
    if(len(data) == 0):
        print("Did not find any photos with extension .jpeg")

        data = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
        print("Trying extension .jpg - Found",len(data),"images :)")
        FLAGS.input_fname_pattern = "*.jpg"


    np.random.shuffle(data)
    # testDatasetSize =  len(data)
    batch_files = data[:min(len(data), testDatasetSize) ]
    sample_files = data[:int(len(data)) ]

    testDataset = [
        get_image(batch_file,
                  input_height=FLAGS.input_height,
                  input_width=FLAGS.input_width,
                  resize_height=FLAGS.output_height,
                  resize_width=FLAGS.output_width,
                  is_crop=FLAGS.is_crop,
                  is_grayscale=False) for batch_file in batch_files]


    print("Loaded dataset",FLAGS.dataset,"Actaull size:",len(data),". Test size:",len(testDataset))



    # pp.pprint(flags.FLAGS.__flags)
    # pp.pprint(flags.FLAGS.__parse)



    newUtils.createFolderName(FLAGS)
    # print(FLAGS.sample_dir)

    newUtils.createConfingCSV(FLAGS.sample_dir,FLAGS)
    # return

    # pp.pprint(flags.FLAGS.__flags)

    # print(FLAGS.sample_dir.default_value)

    # if not os.path.exists(path):
    #     print("path", path, "not found. Creating new folder")
    #     os.makedirs(path)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)




    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True


    print("Starting to load dcgan")

    with tf.Session(config=run_config) as sess:
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


        print(" ")
        print(" ")
        print("GAN 1 -",FLAGS.gan1,"trained for",dcgan.loadTrainingITerations(FLAGS.gan1),"iterations")
        print("GAN 2 -", FLAGS.gan2,"trained for",dcgan.loadTrainingITerations(FLAGS.gan2),"iterations")
        print(" ")

        # print("Loading checkpoint at: ",FLAGS.gan1)
        if not dcgan.loadSilent(FLAGS.gan1):
            raise Exception("[!] Train a model first, then run test mode")
        testScoreGAN_1 = dcgan.evalImages(testDataset,FLAGS,True)
        print(" ")
        print("GAN 1 test score:" , str(testScoreGAN_1) + "%", )

        samplesGAN_1 = dcgan.getGeneratorSamples(sampleDatasetSize,dataset=sample_files,improved_z_noise=False)
        privateSampleScoreGAN_1 = dcgan.evalImages(samplesGAN_1, FLAGS, False)
        print("GAN 1 score on its own samples:", str(privateSampleScoreGAN_1) + "%", )


        # print("Loading checkpoint at: ",FLAGS.gan2)
        if not dcgan.loadSilent(FLAGS.gan2):
            raise Exception("[!] Train a model first, then run test mode")
        testScoreGAN_2 = dcgan.evalImages(testDataset,FLAGS,True)
        print(" ")
        print("GAN 2 test score on test dataset:", str(testScoreGAN_2) + "%", )

        sampleScoreGAN_2 = dcgan.evalImages(samplesGAN_1, FLAGS, False)

        print("GAN 2 score on GAN 1's samples:", str(sampleScoreGAN_2) + "%", )

        samplesGAN_2 = dcgan.getGeneratorSamples(sampleDatasetSize,dataset=sample_files)
        privateSampleScoreGAN_2 = dcgan.evalImages(samplesGAN_2, FLAGS, False)

        print("GAN 2 score on its own samples:", str(privateSampleScoreGAN_2) + "%", )


        testratio = float(testScoreGAN_1) / float(testScoreGAN_2)
        # print(" ")
        # print("testRatio = GAN1",float(testScoreGAN_1) ," / GAN2", float(testScoreGAN_2), " = ",  testratio )


        # print(" ")

        # print("GAN 1 -", FLAGS.gan1,"again")
        # print(" ")

        # print("Loading checkpoint at: ",FLAGS.gan1)
        if not dcgan.loadSilent(FLAGS.gan1):
            raise Exception("[!] Train a model first, then run test mode")

        # print(" ")
        sampleScoreGAN_1 = dcgan.evalImages(samplesGAN_2,FLAGS,False)
        print(" ")
        print("GAN 1 score on GAN 2's samples:", str(sampleScoreGAN_1) + "%", )

        print(" ")
        print(" ")
        print(" ")

        testratio = float(testScoreGAN_1) / float(testScoreGAN_2)
        print(" ")
        print("testRatio = GAN1",float(testScoreGAN_1) ," / GAN2", float(testScoreGAN_2), " = ",  testratio )





        sampleRatio = float(sampleScoreGAN_1) / float(sampleScoreGAN_2)
        # print("sampleRatio = ", float(sampleScoreGAN_1), " / ", float(sampleScoreGAN_2), " = ", sampleRatio)
        print("sampleRatio = GAN1", float(sampleScoreGAN_1), " / GAN2", float(sampleScoreGAN_2), " = ", sampleRatio)
        print(" ")
        if(testratio > 0.9 and testratio < 1.1):
            print("Test ratio PASSED -  ",testratio,"= 1")
        else:
            print("Test ratio FAILED-  ",testratio,"!= 1")


        if(sampleRatio > 0.98 and sampleRatio < 1.0):
            print("GAN 2 is better, but they are very similar -",FLAGS.gan2)
        elif (sampleRatio < 1.02 and sampleRatio > 1.0):
                print("GAN 1 is better, but they are very similar -", FLAGS.gan1)
        elif(sampleRatio < 0.98):
            print("WINNER GAN 2 -",FLAGS.gan2)
        elif(sampleRatio > 1.02):
            print("WINNER GAN 1 -",FLAGS.gan1)
        elif (sampleRatio == 1.00):
            print("TIE - Sample score is 1.0")
        else:
            print("TIE - Sample score of",sampleRatio,"is too close to 1.0")
        print(" ")
        print(" ")

        print("Testing past checkpoints of GAN 1 on the samples from GAN 2")
        dcgan.evalPastCheckpoints(FLAGS.checkpoint_dir,samplesGAN_2,FLAGS)


if __name__ == '__main__':
    tf.app.run()