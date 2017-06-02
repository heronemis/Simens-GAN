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
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
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
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")




flags.DEFINE_boolean("shuffle_data", False, "Shuffle training data before training [False]")
flags.DEFINE_boolean("improved_z_noise", False, "Use Z noise based on training images [False]")
flags.DEFINE_boolean("static_z", False, "Use the Z noise during each epoch of training[False]")
flags.DEFINE_boolean("minibatch_discrimination", False, "Use of Minibatch Discrimination [False]")
flags.DEFINE_integer("tournament_selection", 0, "0 is turned off. 1 will select the best images from a large selection while 2 will select the worst images. [0,1,2]")

flags.DEFINE_string("gan1", "errorMissingGan1Checkpoint", "Checkpoint folder for GAN 1")
flags.DEFINE_string("gan2", "errorMissingGan2Checkpoint", "Checkpoint folder for GAN 2")


FLAGS = flags.FLAGS


def convertDecimalToPerctage(errorrate):
    per = round(errorrate*100,4)
    return str(per)+"%"


def geCifar():
    import pickle
    nb_samples = 50000
    X = np.zeros((nb_samples, 32, 32, 3), dtype=np.float32)

    for i in range(1, 6):
        fpath = os.path.join("data/cifar/" 'data_batch_%d' % i)
        # fpath = "data_batch_1"
        with open(fpath, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
        data = d[b'data']
        labels = d[b'labels']

        bilder = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)

        bilder = (bilder / 128.) - 1

        # data = data.reshape(data.shape[0], 32, 32, 3)
        X[(i - 1) * 10000:i * 10000, :, :, :] = bilder

    return X

def main(_):


    gan1Name = FLAGS.gan1.replace("checkpoint/", "").replace("/media/simen/PLEX/Simens-GAN/checkpoints_augmented_ikea/","").replace("/","")
    gan2Name = FLAGS.gan2.replace("checkpoint/", "").replace("/media/simen/PLEX/Simens-GAN/checkpoints_augmented_ikea/","").replace("/","")

    maxTrainingEpocs = -1

    useTrainingNoiseIfStaticNoise = True
    useTrainingNoiseIfImprovedNoise = True

    if ("static_z" in str(gan2Name)):
        if(useTrainingNoiseIfStaticNoise):
            gan2Name += "witTrainingNoise"
        else:
            gan2Name += "randomNoise"

    if ("improved_z_noise" in str(gan2Name)):
        if(useTrainingNoiseIfImprovedNoise):
            gan2Name += "witTrainingNoise"
        else:
            gan2Name += "randomNoise"

    #maxTrainingEpocs = -1
    maxTrainingEpocs = -1
    if(maxTrainingEpocs >0):
        print("WARNING; Only testing on the first",maxTrainingEpocs," checkpoints!!")
        print("WARNING; Only testing on the first",maxTrainingEpocs," checkpoints!!")
        print("WARNING; Only testing on the first",maxTrainingEpocs," checkpoints!!")
        print("WARNING; Only testing on the first",maxTrainingEpocs," checkpoints!!")
        print("WARNING; Only testing on the first",maxTrainingEpocs," checkpoints!!")

        #gan1Name += "_epoch_0_to" + maxTrainingEpocs + "_only"
        gan2Name += "_epoch_0_to" + str(maxTrainingEpocs) + "_only"


    if( len(gan1Name) > len(gan2Name)):
        print("Are the models in correct order????")
        print("The shorter on is GAN2 but should probalby be GAN1? amarigt?")

        print("Aborting to be safe")
        return

    print(" ")
    print("GAN 1 -", gan1Name)
    print("GAN 2 -", gan2Name)
    print(" ")


    gamFolder = "gamResults/"
    if not os.path.exists(gamFolder):
        os.makedirs(gamFolder)
        print("Folder not found. Creaintg it!")


    fileName = gan1Name + "__vs__"+ gan2Name
    fileName = fileName.replace("/","")

    if  os.path.exists(gamFolder + fileName + ".csv"):
        print(" ")
        print("Results allready excsists!")
        print(fileName)
        print(fileName)
        print(fileName)
        print("Are you sure?")
        print("Are you sure?")
        print("Are you sure?")
        print("Are you sure?")
        print("Aborting to be safe")
        return



    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    print("Processing test data")

    testDatasetSize = 200*FLAGS.batch_size
    sampleDatasetSize = 200 #*FLAGS.batch_size
    if (FLAGS.dataset == "cifar"):
        data = geCifar()
    else:
        if (FLAGS.dataset == "cat"):
            data = glob(os.path.join("./data", "cat/*", "*.jpg"))
        else:
            data = glob(os.path.join("./data", FLAGS.dataset, "*.jpeg"))
        if(len(data) == 0):
            print("Did not find any photos with extension .jpeg")

            data = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))
            print("Trying extension .jpg - Found",len(data),"images :)")
            FLAGS.input_fname_pattern = "*.jpg"


    np.random.shuffle(data)
    # testDatasetSize =  len(data)
    batch_files = data[:min(len(data), testDatasetSize) ]

    np.random.shuffle(data)
    sample_files = data[:int(len(data)) ]

    if (useTrainingNoiseIfStaticNoise):
        np.random.seed(seed=1337)
    #np.random.seed(seed=456456546)
    static_z = np.random.uniform(-1, 1, [len(data), 100]).astype(np.float32)



    if (FLAGS.dataset == "cifar"):
        testDataset = data
    else:
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



    # newUtils.createFolderName(FLAGS)
    # print(FLAGS.sample_dir)

    #newUtils.createConfingCSV(FLAGS.sample_dir,FLAGS)
    # return

    # pp.pprint(flags.FLAGS.__flags)

    # print(FLAGS.sample_dir.default_value)

    # if not os.path.exists(path):
    #     print("path", path, "not found. Creating new folder")
    #     os.makedirs(path)

    # if not os.path.exists(FLAGS.checkpoint_dir):
    #     os.makedirs(FLAGS.checkpoint_dir)
    # if not os.path.exists(FLAGS.sample_dir):
    #     os.makedirs(FLAGS.sample_dir)




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
            sample_dir=FLAGS.sample_dir,
            evalSize=1)






        minIterations = min(dcgan.loadTrainingITerations(FLAGS.gan1), dcgan.loadTrainingITerations(FLAGS.gan2))


        minEpochs= min(dcgan.getNumberOfCheckpoints(FLAGS.gan1), dcgan.getNumberOfCheckpoints(FLAGS.gan2))


        if(maxTrainingEpocs > 0):
            minEpochs = maxTrainingEpocs

        print("The least trained model has",minEpochs,"checkpoints")

        print(" ")
        print("GAN 1 -",gan1Name,"trained for",dcgan.loadTrainingITerations(FLAGS.gan1),"iterations")
        print("GAN 2 -", gan2Name,"trained for",dcgan.loadTrainingITerations(FLAGS.gan2),"iterations")
        print(" ")


        averageTestScoreGAN_1 = 0
        averageTestScoreGAN_2 = 0

        averageSampleScoreGAN_1 = 0
        averageSampleScoreGAN_2 = 0

        listTestScoreGAN_1 = []
        listTestScoreGAN_2 = []

        listSampleScoreGAN_1 = []
        listSampleScoreGAN_2 = []
        resultsThroughout = []



        gan1Wins = 0
        gan2Wins = 0
        ties = 0

        #minEpochs = 2

        for n in range(0,minEpochs):
            print(" ")
            print(" ")
            print(" ")
            print(" ")
            print(" ")
            print("Epoch",n)

            print("GAN 1 -", gan1Name, "trained for", dcgan.loadTrainingITerations(FLAGS.gan1), "iterations")
            print("GAN 2 -", gan2Name, "trained for", dcgan.loadTrainingITerations(FLAGS.gan2), "iterations")


            #Gan 1
            dcgan.loadCloestsCheckpointNumber(FLAGS.gan1,minIterations,n)

            testScoreGAN_1= dcgan.evalImages(testDataset,FLAGS,True)
            averageTestScoreGAN_1 += testScoreGAN_1
            listTestScoreGAN_1.append(testScoreGAN_1)
            print(" ")
            print("GAN 1 test error rate:" , convertDecimalToPerctage(testScoreGAN_1))



            samplesGAN_1 = dcgan.getGeneratorSamples(sampleDatasetSize)


            # GAN 2

            dcgan.loadCloestsCheckpointNumber(FLAGS.gan2, minIterations, n)
            testScoreGAN_2 = dcgan.evalImages(testDataset,FLAGS,True)
            averageTestScoreGAN_2 += testScoreGAN_2
            listTestScoreGAN_2.append(testScoreGAN_2)
            print("GAN 2 test error rate:", convertDecimalToPerctage(testScoreGAN_2) + "", )



            # if (testScoreGAN_2 == 0 and testScoreGAN_1 == 0):
            #     testratio = 1
            if(testScoreGAN_2 == 0):
                testScoreGAN_2 = 0.00000001
            if(testScoreGAN_1 == 0):
                testScoreGAN_1 = 0.00000001
            testratio = float(testScoreGAN_1) / float(testScoreGAN_2)
            print("Test ratio =",testratio)



            sampleScoreGAN_2 = dcgan.evalImages(samplesGAN_1, FLAGS, False)
            averageSampleScoreGAN_2 += sampleScoreGAN_2
            listSampleScoreGAN_2.append(sampleScoreGAN_2)
            print("GAN 2 error rate on GAN 1's samples:", convertDecimalToPerctage(sampleScoreGAN_2) + "", )

            if("improved_z_noise" in str(gan2Name)):

                cifarDataset = False

                if (FLAGS.dataset == "cifar"):
                    cifarDataset = True

                print("improved_z_noise!!!")
                if(useTrainingNoiseIfImprovedNoise):
                    samplesGAN_2 = dcgan.getGeneratorSamples(sampleDatasetSize,sample_files,improved_z_noise=True,cifarDataset=cifarDataset)
                    np.random.shuffle(data)
                    sample_files = data[:int(len(data))]
                else:
                    samplesGAN_2 = dcgan.getGeneratorSamples(sampleDatasetSize,cifarDataset=cifarDataset)

            elif("static_z" in str(gan2Name)):
                print("static_z!!!")
                samplesGAN_2 = dcgan.getGeneratorSamples(sampleDatasetSize,static_z=static_z)
                #np.random.shuffle(static_z)


            else:
                samplesGAN_2 = dcgan.getGeneratorSamples(sampleDatasetSize)


            #GAN1 again

            dcgan.loadCloestsCheckpointNumber(FLAGS.gan1, minIterations, n)
            sampleScoreGAN_1 = dcgan.evalImages(samplesGAN_2,FLAGS,False)
            averageSampleScoreGAN_1 += sampleScoreGAN_1
            listSampleScoreGAN_1.append(sampleScoreGAN_1)
            print(" ")
            print("GAN 1 test error on GAN 2's samples:", convertDecimalToPerctage(sampleScoreGAN_1) + "", )

            print(" ")
            print("testRatio = GAN1",float(testScoreGAN_1) ," / GAN2", float(testScoreGAN_2), " = ",  testratio )





            if (sampleScoreGAN_1 == 0 and sampleScoreGAN_2 == 0):
                sampleRatio = 1
            elif(sampleScoreGAN_2 == 0):
                sampleRatio = 0
            else:
                sampleRatio = float(sampleScoreGAN_1) / float(sampleScoreGAN_2)
            # print("sampleRatio = ", float(sampleScoreGAN_1), " / ", float(sampleScoreGAN_2), " = ", sampleRatio)
            print("sampleRatio = GAN1", float(sampleScoreGAN_1), " / GAN2", float(sampleScoreGAN_2), " = ", sampleRatio)
            print(" ")
            if(round(testratio) == 1):
                print("Test ratio PASSED - round(",testratio,") = 1.0")
            else:
                print("Test ratio FAILED - round(",testratio,") =", round(testratio))

            print(" ")
            print("testRatio =",testratio," ~ ",round(testratio))
            print("sampleRatio =",sampleRatio," ~ ",round(sampleRatio,3))


            sameplRatioRound = round(sampleRatio,3)
            if (round(testratio) != 1):
                print("Tie (test failed)")
                ties += 1
                resultsThroughout.append("tie_test_failed")
            elif (sameplRatioRound > 1.0):
            # if (sampleRatio < 0.999):
                print("WINNER GAN 1 with score", sampleRatio, " - ", gan1Name)
                gan1Wins += 1
                resultsThroughout.append("GAN_1")
                #print(gan1Name+ " score +"+ str(sampleRatioDiff) +" against " + gan1Name+ " score -"+ str(sampleRatioDiff))
            elif (sameplRatioRound < 1):
                print("WINNER GAN 2 with score",sampleRatio," - ", gan2Name)
                gan2Wins += 1
                resultsThroughout.append("GAN_2")
                #print(gan1Name + " score +" + str(sampleRatioDiff) + " against " + gan2Name + " score -" + str(sampleRatioDiff))

            elif (sampleRatio == 1.00):
                print("TIE - Sample score is 1.0")
                ties += 1
                resultsThroughout.append("TIE")
            else:
                print("TIE - Sample score of", sampleRatio, "is too close to 1.0")
                resultsThroughout.append("TIE")
                ties += 1
            print(" ")
            print(" ")

        print("gan1Wins:",gan1Wins)
        print("gan2Wins:",gan2Wins)
        print("ties:",ties)
        print("")

        averageTestScoreGAN_1 = float(averageTestScoreGAN_1) / float(minEpochs)
        averageTestScoreGAN_2 = float(averageTestScoreGAN_2) / float(minEpochs)

        averageSampleScoreGAN_1 = float(averageSampleScoreGAN_1) / float(minEpochs)
        averageSampleScoreGAN_2 = float(averageSampleScoreGAN_2) / float(minEpochs)

        print("averageTestScoreGAN_1:", averageTestScoreGAN_1)
        print("averageTestScoreGAN_2:", averageTestScoreGAN_2)
        print("averageSampleScoreGAN_1:", averageSampleScoreGAN_1)
        print("averageSampleScoreGAN_2:", averageSampleScoreGAN_2)

        if (testScoreGAN_2 == 0):
            testScoreGAN_2 = 0.00000001
        if (testScoreGAN_1 == 0):
            testScoreGAN_1 = 0.00000001
        testratio = float(averageTestScoreGAN_1) / float(averageTestScoreGAN_2)
        print(" ")
        print("testRatio = GAN1", float(averageTestScoreGAN_1), " / GAN2", float(averageTestScoreGAN_2), " = ", testratio)

        if (averageSampleScoreGAN_1 == 0 and averageSampleScoreGAN_2 == 0):
            sampleRatio = 1
        elif (averageSampleScoreGAN_2 == 0):
            sampleRatio = 0
        else:
            sampleRatio = float(averageSampleScoreGAN_1) / float(averageSampleScoreGAN_2)
        # print("sampleRatio = ", float(averageSampleScoreGAN_1), " / ", float(averageSampleScoreGAN_2), " = ", sampleRatio)
        print("sampleRatio = GAN1", float(averageSampleScoreGAN_1), " / GAN2", float(averageSampleScoreGAN_2), " = ", sampleRatio)
        print(" ")
        if (round(testratio) == 1):
            print("Test ratio PASSED - round(", testratio, ") = 1.0")
        else:
            print("Test ratio FAILED - round(", testratio, ") =", round(testratio))

        print(" ")
        print("testRatio =", testratio, " ~ ", round(testratio))
        print("sampleRatio =", sampleRatio, " ~ ", round(sampleRatio, 3))

        sameplRatioRound = round(sampleRatio, 3)
        winningGan = "tie"

        if (round(testratio) != 1):
            print("Tie (test failed)")
            # ties += 1
            winningGan = "tie_test_failed"
        elif (sameplRatioRound > 1.0):
            # if (sampleRatio < 0.999):
            print("WINNER GAN 1 with score", sampleRatio, " - ", gan1Name)
            # gan1Wins += 1
            winningGan = "GAN_1"
            # print(gan1Name+ " score +"+ str(sampleRatioDiff) +" against " + gan1Name+ " score -"+ str(sampleRatioDiff))
        elif (sameplRatioRound < 1):
            print("WINNER GAN 2 with score", sampleRatio, " - ", gan2Name)
            # gan2Wins += 1
            winningGan = "GAN_2"
            # print(gan1Name + " score +" + str(sampleRatioDiff) + " against " + gan2Name + " score -" + str(sampleRatioDiff))

        elif (sampleRatio == 1.00):
            print("TIE - Sample score is 1.0")
            # ties += 1
        else:
            print("TIE - Sample score of", sampleRatio, "is too close to 1.0")
            # ties += 1
        print(" ")

        newUtils.createGAMCSV(resultsThroughout,gan1Name,gan2Name,listTestScoreGAN_1, listTestScoreGAN_2, listSampleScoreGAN_1, listSampleScoreGAN_2,winningGan,sampleRatio,testratio,gan1Wins,gan2Wins,ties,testDatasetSize,sampleDatasetSize*FLAGS.batch_size)

        print(" ")


















        return


        #####################################
        gan1Name = FLAGS.gan1.replace("checkpoint/","")
        gan2Name = FLAGS.gan2.replace("checkpoint/","")


        minIterations = min(dcgan.loadTrainingITerations(FLAGS.gan1), dcgan.loadTrainingITerations(FLAGS.gan2))
        #minIterations = 53947
        print("Minimum training iterations are",minIterations)

        print(" ")
        print(" ")
        print("GAN 1 -",gan1Name,"trained for",dcgan.loadTrainingITerations(FLAGS.gan1),"iterations")
        print("GAN 2 -", gan2Name,"trained for",dcgan.loadTrainingITerations(FLAGS.gan2),"iterations")
        print(" ")

        # print("Loading checkpoint at: ",FLAGS.gan1)
        if not dcgan.loadCloestsCheckpoint(FLAGS.gan1,minIterations):
            raise Exception("[!] Train a model first, then run test mode")
        testScoreGAN_1 = dcgan.evalImages(testDataset,FLAGS,True)
        print(" ")
        print("GAN 1 test error rate:" , convertDecimalToPerctage(testScoreGAN_1))

        samplesGAN_1 = dcgan.getGeneratorSamples(sampleDatasetSize,dataset=sample_files,improved_z_noise=False)
        privateSampleScoreGAN_1 = dcgan.evalImages(samplesGAN_1, FLAGS, False)
        print("GAN 1 error rate on its own samples:", convertDecimalToPerctage(privateSampleScoreGAN_1))


        # print("Loading checkpoint at: ",FLAGS.gan2)
        if not dcgan.loadCloestsCheckpoint(FLAGS.gan2,minIterations):
            raise Exception("[!] Train a model first, then run test mode")
        testScoreGAN_2 = dcgan.evalImages(testDataset,FLAGS,True)
        print(" ")
        print("GAN 2 test error rate:", convertDecimalToPerctage(testScoreGAN_2) + "", )

        sampleScoreGAN_2 = dcgan.evalImages(samplesGAN_1, FLAGS, False)

        print("GAN 2 error rate on GAN 1's samples:", convertDecimalToPerctage(sampleScoreGAN_2) + "", )

        samplesGAN_2 = dcgan.getGeneratorSamples(sampleDatasetSize,dataset=sample_files,improved_z_noise=True)
        privateSampleScoreGAN_2 = dcgan.evalImages(samplesGAN_2, FLAGS, False)

        print("GAN 2 error rate on its own samples:", convertDecimalToPerctage(privateSampleScoreGAN_2) + "", )


        # print(" ")
        # print("testRatio = GAN1",float(testScoreGAN_1) ," / GAN2", float(testScoreGAN_2), " = ",  testratio )


        # print(" ")

        # print("GAN 1 -", FLAGS.gan1,"again")
        # print(" ")

        # print("Loading checkpoint at: ",FLAGS.gan1)
        if not dcgan.loadCloestsCheckpoint(FLAGS.gan1,minIterations):
            raise Exception("[!] Train a model first, then run test mode")

        # print(" ")
        sampleScoreGAN_1 = dcgan.evalImages(samplesGAN_2,FLAGS,False)
        print(" ")
        print("GAN 1 test error on GAN 2's samples:", convertDecimalToPerctage(sampleScoreGAN_1) + "", )

        print(" ")
        print(" ")
        print(" ")

        if(testScoreGAN_2 == 0):
            testratio = 0
        else:
            testratio = float(testScoreGAN_1) / float(testScoreGAN_2)
        print(" ")
        print("testRatio = GAN1",float(testScoreGAN_1) ," / GAN2", float(testScoreGAN_2), " = ",  testratio )






        sampleRatio = float(sampleScoreGAN_1) / float(sampleScoreGAN_2)
        # print("sampleRatio = ", float(sampleScoreGAN_1), " / ", float(sampleScoreGAN_2), " = ", sampleRatio)
        print("sampleRatio = GAN1", float(sampleScoreGAN_1), " / GAN2", float(sampleScoreGAN_2), " = ", sampleRatio)
        print(" ")
        if(round(testratio) == 1):
            print("Test ratio PASSED - round(",testratio,") = 1.0")
        else:
            print("Test ratio FAILED - round(",testratio,") =", round(testratio))

        print(" ")
        print("testRatio =",round(testratio,3))
        print("sampleRatio =",round(sampleRatio,3))


        sampleRatioDiff = abs(float(sampleScoreGAN_1) - float(sampleScoreGAN_2))

        if (sampleRatio < 0.999):
            print("WINNER GAN 1 with score", sampleRatio, " - ", gan1Name)
            #print(gan1Name+ " score +"+ str(sampleRatioDiff) +" against " + gan1Name+ " score -"+ str(sampleRatioDiff))
        elif (sampleRatio > 1.001):
            print("WINNER GAN 2 with score",sampleRatio," - ", gan2Name)
            #print(gan1Name + " score +" + str(sampleRatioDiff) + " against " + gan2Name + " score -" + str(sampleRatioDiff))

        elif (sampleRatio == 1.00):
            print("TIE - Sample score is 1.0")
        else:
            print("TIE - Sample score of", sampleRatio, "is too close to 1.0")
        print(" ")
        print(" ")

        print(" ")
        print("GAN 1 -",FLAGS.gan1,"trained for",dcgan.loadTrainingITerations(FLAGS.gan1),"iterations")
        print("GAN 2 -", FLAGS.gan2,"trained for",dcgan.loadTrainingITerations(FLAGS.gan2),"iterations")
        print(" ")



        print("Testing past checkpoints of GAN 1 on the samples from GAN 2")
        sampleScoreGAN_1,testScoreGAN_1 = dcgan.evalPastCheckpoints(FLAGS.gan1,testDataset,samplesGAN_2,FLAGS,maxIterationNumber=minIterations)

        print(" ")
        print(" ")

        print("Testing past checkpoints of GAN 2 on the samples from GAN 1")
        sampleScoreGAN_2, testScoreGAN_2 = dcgan.evalPastCheckpoints(FLAGS.gan2,testDataset, samplesGAN_1, FLAGS,maxIterationNumber=minIterations)





        ####
        if(testScoreGAN_2 == 0):
            testratio = 0
        else:
            testratio = float(testScoreGAN_1) / float(testScoreGAN_2)
        print(" ")
        print("testRatio = GAN1",float(testScoreGAN_1) ," / GAN2", float(testScoreGAN_2), " = ",  testratio )






        sampleRatio = float(sampleScoreGAN_1) / float(sampleScoreGAN_2)
        # print("sampleRatio = ", float(sampleScoreGAN_1), " / ", float(sampleScoreGAN_2), " = ", sampleRatio)
        print("sampleRatio = GAN1", float(sampleScoreGAN_1), " / GAN2", float(sampleScoreGAN_2), " = ", sampleRatio)
        print(" ")
        if(round(testratio) == 1):
            print("Test ratio PASSED - round(",testratio,") = 1.0")
        else:
            print("Test ratio FAILED - round(",testratio,") =", round(testratio))


        print(" ")
        print("testRatio =",round(testratio,3))
        print("sampleRatio =",round(sampleRatio,3))


        sampleRatioDiff = abs(float(sampleScoreGAN_1) - float(sampleScoreGAN_2))

        if (sampleRatio < 0.999):
            print("WINNER GAN 1 with score", sampleRatio, " - ", gan1Name)
            #print(gan1Name+ " score +"+ str(sampleRatioDiff) +" against " + gan1Name+ " score -"+ str(sampleRatioDiff))
        elif (sampleRatio > 1.001):
            print("WINNER GAN 2 with score",sampleRatio," - ", gan2Name)
            #print(gan1Name + " score +" + str(sampleRatioDiff) + " against " + gan2Name + " score -" + str(sampleRatioDiff))

        elif (sampleRatio == 1.00):
            print("TIE - Sample score is 1.0")
        else:
            print("TIE - Sample score of", sampleRatio, "is too close to 1.0")
        print(" ")
        print(" ")


if __name__ == '__main__':
    tf.app.run()