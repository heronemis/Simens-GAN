from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *
from newUtils import *

import PIL
from PIL import Image
from PIL import ImageOps
from random import randint

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, evalSize=1):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.secondDicsriminator = False
        if (self.secondDicsriminator):
            print("secondDicsriminator is True")

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.test = False

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.evalSize = batch_size * evalSize

        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if self.secondDicsriminator:
            self.d2_bn1 = batch_norm(name='d2_bn1')
            self.d2_bn2 = batch_norm(name='d2_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')
            if self.secondDicsriminator:
                self.d2_bn3 = batch_norm(name='d2_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
            self.y_eval = tf.placeholder(tf.float32, [self.evalSize, self.y_dim], name='y_eval')
            self.y_eval2 = tf.placeholder(tf.float32, [self.evalSize * 2, self.y_dim], name='y_eval2')

        if self.is_crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_height, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        self.inputsArchiveFake = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='fake_fromArchive')

        self.inputsProc = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='inputsProc')

        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        self.eval_input = tf.placeholder(tf.float32, [self.evalSize] + image_dims, name='eval_input')

        inputs = self.inputs
        inputsArchiveFake = self.inputsArchiveFake
        inputsProc = self.inputsProc
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits = \
                self.discriminator(inputs, self.y, reuse=False)

            self.evaulator = self.evaluators(self.z, self.y_eval)
            self.sampler = self.sampler(self.z, self.y)

            self.D_, self.D_logits_ = \
                self.discriminator(self.G, self.y, reuse=True)

            self.discriminatorOutput = self.discriminator(inputs, self.y, reuse=True)
            self.discriminatorEval = self.discriminator(self.eval_input, self.y_eval, reuse=True,
                                                        tempBatchSize=self.evalSize)
        else:

            self.D, self.D_logits = self.discriminator(inputs)

            if self.secondDicsriminator:
                self.D2, self.D2_logits = self.discriminator(inputs, secondDiscriminator=True)
                # self.D2, self.D2_logits = self.discriminator(inputsArchiveFake,secondDiscriminator=True)

            # self.GInit = self.generator(self.z, useBatching=False)
            self.G = self.generator(self.z)

            self.generatorOuput = self.generator(self.z, y=None, reuse=True, tempBatchSize=self.evalSize,
                                                 useBatching=False)

            self.discriminatorOutput = self.discriminator(inputs, reuse=True)

            # self.G = self.generator(self.z,reuse=True,useBatching=True)
            # self.generatorEval = self.generator(self.z, y=None, reuse=True, tempBatchSize=self.evalSize)






            self.sampler = self.sampler(self.z)

            # self.GBatch = self.batchGenerator() #test=False
            # self.GBatch = self.batchGenerator(self.z) #test=False
            # self.generatorEvalasdasdasd = self.generator(self.z, y=None, reuse=True, tempBatchSize=self.evalSize,test=True)

            # self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

            if self.secondDicsriminator:
                self.D2_, self.D2_logits_ = self.discriminator(self.inputsArchiveFake, reuse=True,
                                                               secondDiscriminator=True)
                # self.D2Archive, self.D2_logitsArchive = self.discriminator(self.G, reuse=True,secondDiscriminator=True)
            #
            # self.D_, self.D_logits_ = self.discriminator(inputsProc, reuse=True)
            self.generatorEval = self.generator(self.z, y=None, reuse=True, tempBatchSize=self.evalSize,
                                                useBatching=False)



            # self.GTest = self.generator(self.z,reuse=True, useBatching=True)



            # self.generatorEval = self.generator(self.z, y=None, reuse=True, tempBatchSize=self.evalSize, useBatching=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)

        if self.secondDicsriminator:
            self.d2_sum = histogram_summary("d2", self.D2)
            self.d2__sum = histogram_summary("d2_", self.D2_)

        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        if self.secondDicsriminator:
            self.d2_loss_real = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D2_logits, tf.ones_like(self.D2)))
            self.d2_loss_fake = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.zeros_like(self.D2_)))

            self.g_loss_normal = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))


            self.g_loss_historic = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.ones_like(self.D2_)))

            self.g_loss = self.g_loss_normal + self.g_loss_historic
            # self.g2_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D2_logitsArchive, tf.ones_like(self.D2_)))

        # self.d_loss_real = tf.reduce_mean(
        #     sigmoid_cross_entropy_with_logits(self.D_logits, tf.zeros_like(self.D)))
        # self.d_loss_fake = tf.reduce_mean(
        #     sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        # self.g_loss = tf.reduce_mean(
        #     sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake



        if self.secondDicsriminator:
            self.d2_loss_real_sum = scalar_summary("d2_loss_real", self.d2_loss_real)
            self.d2_loss_fake_sum = scalar_summary("d2_loss_fake", self.d2_loss_fake)

            self.d2_loss = self.d2_loss_real + self.d2_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
        if self.secondDicsriminator:
            #self.g2_loss_sum = scalar_summary("g2_loss", self.g2_loss)
            self.d2_loss_sum = scalar_summary("d2_loss", self.d2_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        if self.secondDicsriminator:
            self.d2_vars = [var for var in t_vars if 'd2_' in var.name]

        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=50)

    # def evalImages(self, lastAccuracy, evalDataset, config, realImages=True):
    def evalImages(self, evalDataset, config, realImages=True, useErrorRate=True):

        batch_idxs = min(len(evalDataset), config.train_size) // config.batch_size

        correct = 0
        correctScale = 0

        for idx in xrange(0, batch_idxs):
            batch_images = evalDataset[idx * config.batch_size:(idx + 1) * config.batch_size]
            discriminatorScoresBatch = self.sess.run([self.discriminatorOutput],
                                                     feed_dict={self.inputs: batch_images})

            for res in range(0, config.batch_size):
                # print("Real ", evalDiscOuput[0][0][res],' - ', 1.0)
                if (realImages):
                    if (discriminatorScoresBatch[0][0][res] > 0.5):
                        correct += 1
                    if (discriminatorScoresBatch[0][0][res] > 0.5):
                        correctScale += 1.0 - (1.0 - discriminatorScoresBatch[0][0][res])
                else:
                    if (discriminatorScoresBatch[0][0][res] < 0.5):
                        correct += 1
                        # print("     - correct!")

        # if (realImages):
        #     percentageScale = ((float(correctScale) / float(len(evalDataset))))
        #     # print("Error rate scale:",percentageScale,"%")

        if (useErrorRate):
            percentage = ((float(correct) / float(len(evalDataset))))
            return percentage
        else:
            print("Not using multiple outputs")

        percentage = ((float(correct) / float(len(evalDataset))) * 100)
        #
        deri = ""
        # if (percentage > lastAccuracy):
        #     deri = "++"
        # elif (percentage < lastAccuracy):
        #     deri = "--"
        #
        # if (realImages):
        #     print("Accuracy(REAL):", deri + str(percentage) + "% (", correct, "samples )")
        # else:
        #     print("Accuracy(FAKE):", deri + str(percentage) + "% (", correct, "samples )")
        return percentage

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
            # data_X, data_y = self.load_mnist()
            data_X, data_y, testing_x_all, testing_y_all = self.load_mnist_with_test()
        else:
            if (config.dataset == "cat"):
                data = glob(os.path.join("./data", "cat/*", "*.jpg"))
            elif (config.dataset == "cifar"):
                data = self.geCifar()
            else:
                data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))

        ## SIMENS LILLE CONFING ##
        self.train_size = config.train_size
        shouldLoadData = True
        useEvalSet = True
        writeLogs = False
        tournament_selection_noise = config.tournament_selection > 0
        useImproved_z_noise = config.improved_z_noise
        useStaticZNoise = config.static_z
        auditionNoiseGeneratorOnly = config.tournment_noise_g_only

        initCSV(config.sample_dir)

        configString = getAddons(config)

        useSampleArchive = config.multiGanMode > 0

        if(useSampleArchive):
            usedIndecies = []
            sampleArchive = np.zeros((self.batch_size*10, self.output_height, self.output_width, self.c_dim),
                                  dtype=np.float32)

        ## SIMENS LILLE CONFING ##


        if (useStaticZNoise):
            np.random.seed(seed=1337)
            static_z = np.random.uniform(-1, 1, [len(data), self.z_dim]).astype(np.float32)

        elif (useImproved_z_noise):
            print("Greating all Z-noise based on training data. Might take some time")
            static_improved_z = np.random.uniform(-1, 1, [len(data), self.z_dim]).astype(np.float32)

            basewidth = 10
            indexCounter = 0
            for imgName in data:
                if (config.dataset == "cifar"):
                    #img = imgName
                    # print(imgName[0][0])

                    intData = (imgName +1) * 128.
                    rawData = np.array(intData, np.uint8)
                    # print(intData[0][0])
                    #intData = imgName.reshape(256,256,3)
                    img = Image.fromarray(rawData)

                else:
                    img = Image.open(imgName)
                # img = img.convert('L')  # convert image to greyscale
                wpercent = (basewidth / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)  # resizes the image to 10x10
                img = img.convert('L')  # convert image to black and white
                # name = 'processedImages/' + str(counter) + '.jpeg'
                # img = ImageOps.invert(img)
                # img.save(name)

                pix = np.array(img, np.float32)
                pix = (pix - 128) / 128  # Scales the pixels to be between -1 and 1
                pix = pix.flatten()  # flattens the image to a single array of lenght of 100

                static_improved_z[indexCounter] = pix  # Adds the image to the z-array
                indexCounter += 1
                if (indexCounter == len(data) / 2):
                    print("Only halfway......")

        if (useEvalSet):
            if (config.dataset == "cifar"):
                eval_real = self.geCifar()[0:self.evalSize]
            else:
                eval_files = data[0:self.evalSize]
                evalSamplesReal = [
                    get_image(sample_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              is_crop=self.is_crop,
                              is_grayscale=self.is_grayscale) for sample_file in eval_files]
                if (self.is_grayscale):
                    eval_real = np.array(evalSamplesReal).astype(np.float32)[:, :, :, None]
                else:
                    eval_real = np.array(evalSamplesReal).astype(np.float32)

        if (config.shuffle_data):
            np.random.shuffle(data)
            print("Shuffling trainingdata")

        d_optim = tf.train.AdamOptimizer(config.learning_rate_D, beta1=config.beta1_D) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate_G, beta1=config.beta1_G) \
            .minimize(self.g_loss, var_list=self.g_vars)

        if self.secondDicsriminator:
            d2_optim = tf.train.AdamOptimizer(config.learning_rate_D, beta1=config.beta1_D) \
                .minimize(self.d2_loss, var_list=self.d2_vars)

            #g2_optim = tf.train.AdamOptimizer(config.learning_rate_G, beta1=config.beta1_G) \
             #   .minimize(self.g2_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # self.G = self.generator(self.z, reuse=True, useBatching=True)

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        if self.secondDicsriminator:
            self.d2_sum = merge_summary(
                [self.d2_sum, self.d2_loss_real_sum, self.d2_loss_sum])


            #self.g_sum2 = merge_summary([self.z_sum, self.d2__sum, self.G_sum, self.d2_loss_fake_sum, self.g_loss_sum])

        # self.GTest = self.generator(self.z, reuse=True, useBatching=True)





        if (writeLogs):
            self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        lastFakeAccuracy = 0
        lastRealAccuracy = 0

        self.test = True

        if config.dataset == 'mnist':
            sample_inputs = data_X[0:self.sample_num]
            sample_labels = data_y[0:self.sample_num]

            testing_x = testing_x_all[0:self.evalSize]
            testing_y = testing_y_all[0:self.evalSize]

        elif (config.dataset == "cifar"):
            sample_inputs = data[0:self.sample_num]
        else:
            sample_files = data[0:self.sample_num]
            sample = [
                get_image(sample_file,
                          input_height=self.input_height,
                          input_width=self.input_width,
                          resize_height=self.output_height,
                          resize_width=self.output_width,
                          is_crop=self.is_crop,
                          is_grayscale=self.is_grayscale) for sample_file in sample_files]
            if (self.is_grayscale):
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if (not shouldLoadData):
            print(" [x] Not loading data")

        elif self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(data_X), config.train_size) // config.batch_size
            else:
                if (config.dataset == "cat"):
                    data = glob(os.path.join("./data", "cat/*", "*.jpg"))
                elif (config.dataset == "cifar"):
                    _ = 10
                else:
                    data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
                batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_labels = data_y[idx * config.batch_size:(idx + 1) * config.batch_size]

                elif (config.dataset == "cifar"):
                    batch_images = data[idx * config.batch_size:(idx + 1) * config.batch_size]



                else:
                    batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch = [
                        get_image(batch_file,
                                  input_height=self.input_height,
                                  input_width=self.input_width,
                                  resize_height=self.output_height,
                                  resize_width=self.output_width,
                                  is_crop=self.is_crop,
                                  is_grayscale=self.is_grayscale) for batch_file in batch_files]
                    if (self.is_grayscale):
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                if (tournament_selection_noise):

                    _, batch_z = self.batchGenerator(config.tournament_selection)

                elif (useStaticZNoise):
                    batch_z = static_z[idx * config.batch_size:(idx + 1) * config.batch_size]

                elif (useImproved_z_noise):

                    batch_z = static_improved_z[idx * config.batch_size:(idx + 1) * config.batch_size]

                    # basewidth = 10
                    # indexCounter = 0
                    # for imgName in batch_files:
                    #     img = Image.open(imgName)
                    #     # img = img.convert('L')  # convert image to greyscale
                    #     wpercent = (basewidth / float(img.size[0]))
                    #     hsize = int((float(img.size[1]) * float(wpercent)))
                    #     img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)  # resizes the image to 10x10
                    #     img = img.convert('L')  # convert image to black and white
                    #     # name = 'processedImages/' + str(counter) + '.jpeg'
                    #     # img = ImageOps.invert(img)
                    #     # img.save(name)
                    #
                    #     pix = np.array(img, np.float32)
                    #     pix = (pix - 128) / 128  # Scales the pixels to be between -1 and 1
                    #     pix = pix.flatten()  # flattens the image to a single array of lenght of 100
                    #
                    #     batch_z[indexCounter] = pix  # Adds the image to the z-array
                    #     indexCounter += 1

                # else:
                #     batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)


                if config.dataset == 'mnist':
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                   })
                    if (writeLogs):
                        self.writer.add_summary(summary_str, counter)

                    if (lastRealAccuracy > 70 or True):
                        # Update G network
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                       feed_dict={
                                                           self.z: batch_z,
                                                           self.y: batch_labels,
                                                       })
                        if (writeLogs):
                            self.writer.add_summary(summary_str, counter)

                        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                        _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                       feed_dict={self.z: batch_z, self.y: batch_labels})

                        if (writeLogs):
                            self.writer.add_summary(summary_str, counter)
                    else:
                        print("Freezing the generator!")

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels
                    })
                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                else:
                    # Update D network
                    # if (lastRealAccuracy > 70):
                    # _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict={self.inputs: batch_images, self.inputsProc: inputs})

                    if (not auditionNoiseGeneratorOnly):
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                       feed_dict={self.inputs: batch_images, self.z: batch_z})
                    else:
                        print("Discirminator is getting random noise")
                        random_batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                       feed_dict={self.inputs: batch_images, self.z: random_batch_z})

                    if self.secondDicsriminator:
                        if(useSampleArchive):
                            np.random.shuffle(sampleArchive)
                            batch_images_archive = sampleArchive[0 * config.batch_size:(0 + 1) * config.batch_size]

                            _, summary_str = self.sess.run([d2_optim, self.d2_sum], feed_dict={self.inputs: batch_images,
                                                                                               self.inputsArchiveFake: batch_images_archive})

                    if (writeLogs):
                        self.writer.add_summary(summary_str, counter)

                    if (lastRealAccuracy > 70 or True):
                        # Update G network
                        # _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={self.z: batch_z,self.inputsArchiveFake: batch_images_archive})
                        # if (writeLogs):
                        #     self.writer.add_summary(summary_str, counter)

                        if self.secondDicsriminator:
                            if (useSampleArchive):
                                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                               feed_dict={self.z: batch_z,
                                                                          self.inputsArchiveFake: batch_images_archive})


                                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                               feed_dict={self.z: batch_z,self.inputsArchiveFake: batch_images_archive})
                        else:
                            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                           feed_dict={self.z: batch_z})
                            if (writeLogs):
                                self.writer.add_summary(summary_str, counter)


                            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                           feed_dict={self.z: batch_z})
                            if (writeLogs):
                                self.writer.add_summary(summary_str, counter)
                    else:
                        print("Not updating generator network")

                    if self.secondDicsriminator:
                        if (useSampleArchive):
                            errD_fake = self.d_loss_fake.eval({self.z: batch_z,self.inputsArchiveFake: batch_images_archive})
                            errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                            errG = self.g_loss.eval({self.z: batch_z,self.inputsArchiveFake: batch_images_archive})

                    else:
                        errD_fake = self.d_loss_fake.eval(
                            {self.z: batch_z})
                        errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                        errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                # print("Creating eval data")


                # if idx <= 100 or True:
                if np.mod(counter, 50) == 1:

                    print("self.evalSize=",self.evalSize)

                    eval_z = np.random.uniform(-1, 1, [self.evalSize, self.z_dim]).astype(np.float32)
                    samples_eval = self.sess.run(
                        [self.generatorEval],
                        feed_dict={
                            self.z: eval_z,
                        }
                    )

                    samples_eval = np.asarray(samples_eval[0])

                    lastRealAccuracy = self.evalImages(eval_real, config, realImages=True, useErrorRate=False)
                    lastFakeAccuracy = self.evalImages(samples_eval, config, realImages=False, useErrorRate=False)

                    iterationNumber = float(idx) / float(batch_idxs) + float(epoch)

                    writeAccuracyToFile(config.sample_dir,
                                        [iterationNumber, lastRealAccuracy / 100.0, lastFakeAccuracy / 100.0,
                                         (lastRealAccuracy + lastFakeAccuracy) / 200.00])

                if np.mod(counter, 50) == 1 and useSampleArchive:

                    archive_z = np.random.uniform(-1, 1, [64, self.z_dim]).astype(np.float32)

                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: archive_z,
                            self.inputs: sample_inputs, self.inputsArchiveFake: batch_images_archive
                        },
                    )
                    imagesToReplace = randint(1, 16)
                    for i in range(0, imagesToReplace):
                        randomInxed = randint(0, len(sampleArchive)-1)

                        if(randomInxed not in usedIndecies):
                            usedIndecies.append(randomInxed)

                        sampleArchive[randomInxed] = samples[i]
                    print("Indecies used is", len(usedIndecies),"/",len(sampleArchive))
                    if(len(usedIndecies) == len(sampleArchive)):
                        usedIndecies = []



                if np.mod(counter, 50) == 1:

                    # self.batchGenerator()

                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }
                        )

                        # print("Creating eval data")
                        samples_eval = self.sess.run(
                            [self.evaulator],
                            feed_dict={
                                self.z: eval_z,
                                # self.eval_input : testing_x,
                                self.y_eval: testing_y,
                            }
                        )

                        samples_eval = np.asarray(samples_eval[0])
                        # print("Done. Size: ", len(samples_eval))
                        # print("Done. Size[0]: ", len(samples_eval[0]))
                        # print("Done. Size[0][0]: ", len(samples_eval[0][0]))


                        labelsReal = [1] * self.evalSize
                        labelsFake = [0] * self.evalSize

                        labels = np.asarray(labelsReal + labelsFake)

                        # print("testing_x.shape",testing_x.shape)
                        # print("testing_x.shape",samples_eval.shape)

                        # X = np.concatenate((testing_x,samples_eval ), axis=0)
                        # yLAbels = np.concatenate((testing_y, testing_y), axis=0).astype(np.int)
                        #
                        # seed = 547
                        # np.random.seed(seed)
                        # np.random.shuffle(X)
                        # np.random.seed(seed)
                        # np.random.shuffle(yLAbels)
                        # np.random.seed(seed)
                        # np.random.shuffle(labels)

                        # print(labels)



                        rho = np.zeros((64, 28, 28, 1))

                        for i in range(0, 32):
                            samples[i * 2] = batch_images[i * 2]
                            # samples[i*2] = batch_images[i]
                        batchOutput = self.sess.run([self.discriminatorOutput],
                                                    feed_dict={self.inputs: samples, self.y: batch_labels})

                        evalDiscOuput = self.sess.run([self.discriminatorEval],
                                                      feed_dict={self.eval_input: samples_eval, self.y_eval: testing_y})

                        correct = 0
                        for res in range(0, self.evalSize):
                            # print("Real ", evalDiscOuput[0][0][res],' - ', 1.0)
                            if (evalDiscOuput[0][0][res] < 0.5):
                                correct += 1
                                # print("     - correct!")
                        percentage = ((float(correct) / float(self.evalSize)) * 100)

                        deri = ""
                        if (percentage > lastRealAccuracy):
                            deri = "+"
                        elif (percentage < lastRealAccuracy):
                            deri = "-"

                        print("Correct (Real): ", deri, percentage, "% (", correct, ")")

                        lastRealAccuracy = percentage

                        evalDiscOuput = self.sess.run([self.discriminatorEval],
                                                      feed_dict={self.eval_input: testing_x, self.y_eval: testing_y})

                        correct = 0
                        for res in range(0, self.evalSize):
                            # print("Real ", evalDiscOuput[0][0][res],' - ', 1.0)
                            if (evalDiscOuput[0][0][res] > 0.5):
                                correct += 1
                                # print("     - correct!")

                        percentage = ((float(correct) / float(self.evalSize)) * 100)

                        deri = ""
                        if (percentage > lastFakeAccuracy):
                            deri = "+"
                        elif (percentage < lastFakeAccuracy):
                            deri = "-"

                        print("Correct (Fake): ", deri, percentage, "% (", correct, ")")

                        lastFakeAccuracy = percentage

                        # # rho[0] = samples[0]
                        # scoreTest, d_loss, g_loss = self.sess.run(
                        #     [self.discriminator, self.d_loss, self.g_loss],
                        #     feed_dict={
                        #         self.inputs : rho,#samples[0],
                        #         self.y: sample_labels,
                        #     }
                        # )
                        # print("Scores: ",batchOutput[0][0])
                        # print("batch_labels: ", np.argmax(batch_labels))
                        # print(" - ")
                        # print("Scores: ", len(batchOutput))
                        # print(" - ")


                        # batchSamples = []
                        # for i in range(0, 64):
                        #     batchSamples.append( np.argmax(batch_labels[i]) )





                        save_images(samples, [8, 8],
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx),
                                    batchOutput[0][0])
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                        # return
                    else:

                        samples = None

                        if (tournament_selection_noise and False):
                            getBestImagesOnly = config.tournament_selection == 1
                            samples, sample_z = self.batchGenerator(config.tournament_selection)

                        elif (useStaticZNoise):
                            sample_z = static_z[idx * config.batch_size:(idx + 1) * config.batch_size]


                        elif (useImproved_z_noise):
                            # sample_z = batch_z
                            sample_z = static_improved_z[idx * config.batch_size:(idx + 1) * config.batch_size]
                            print("Using improved noise for sample generation")

                        else:
                            sample_z = np.random.uniform(-1, 1, [64, self.z_dim]).astype(np.float32)

                        #
                        # eval_z = np.random.uniform(-1, 1, [64, self.z_dim]).astype(np.float32)
                        # samples, eval_z = self.batchGenerator()

                        if (samples is None):
                            if self.secondDicsriminator:
                                if (useSampleArchive):
                                    samples, d_loss, g_loss = self.sess.run(
                                        [self.sampler, self.d_loss, self.g_loss],
                                        feed_dict={
                                            self.z: sample_z,
                                            self.inputs: sample_inputs, self.inputsArchiveFake: batch_images_archive
                                        },
                                    )
                            else:
                                samples, d_loss, g_loss = self.sess.run(
                                    [self.sampler, self.d_loss, self.g_loss],
                                    feed_dict={
                                        self.z: sample_z,
                                        self.inputs: sample_inputs,
                                    },
                                )


                        # for i in range(0, 64):
                        #     randomInxed = randint(0,len(sampleArchive))
                        #     sampleArchive[randomInxed] = samples[i]

                        # samples = self.sess.run(
                        #     [self.generatorEval],
                        #     feed_dict={
                        #         self.z: eval_z,
                        #     }
                        # )


                        # samples = self.sess.run(
                        #     [self.G],
                        #     feed_dict={
                        #         self.z: eval_z,
                        #     }
                        # )

                        # print("Done. Size: ", len(samples))
                        # print("Done. Size[0]: ", len(samples[0]))
                        # print("Done. Size[0][0]: ", len(samples[0][0]))
                        # print("Done. Size[0][0]: ", len(samples[0][0][0]))
                        # print("Done. Size[0][0]: ", len(samples[0][0][0][0]))




                        # samples = np.asarray(samples[0])
                        for i in range(0, 32):
                            samples[i * 2] = batch_images[i * 2]
                            # samples[i*2] = batch_images[i]

                        batchOutput = self.sess.run([self.discriminatorOutput],
                                                    feed_dict={self.inputs: samples})

                        # evalDiscOuput = self.sess.run([self.discriminatorEval],
                        #                               feed_dict={self.eval_input: samples_eval, self.y_eval: testing_y})
                        #
                        #
                        # # try:
                        # samples, d_loss, g_loss = self.sess.run(
                        #     [self.sampler, self.d_loss, self.g_loss],
                        #     feed_dict={
                        #         self.z: sample_z,
                        #         self.inputs: sample_inputs,
                        #     },
                        # )
                        save_images(samples, [8, 8],
                                    './{}/{}_{:02d}_{:04d}{}.png'.format(config.sample_dir, config.dataset, epoch, idx,
                                                                         configString), batchOutput[0][0])  #
                        # print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                        # except:
                        #     print("one pic error!...")

                        # if np.mod(counter, 2000) == 1 or np.mod(counter-1, 2000) == 1:
                        #     print("Saving checkpoint")
                        #     self.save(config.checkpoint_dir, counter)
            print("End of current Epoch, saving checkpoint")
            self.save(config.checkpoint_dir, counter)

    def getGeneratorSamples(self, sampleSize, dataset=None, improved_z_noise=False, static_z=None, cifarDataset=False):

        # sampleSize = 200
        # if (dataset != None):
        #     sampleSize = len(dataset)


        if (improved_z_noise):
            print("improved_z_noise is turned on!")
            print("improved_z_noise is turned on!")
            print("improved_z_noise is turned on!")
            print("improved_z_noise is turned on!")

        if (static_z != None):
            print("static_z=static_z")
            print("static_z=static_z")
            print("static_z=static_z")
            print("static_z=static_z")
            print("static_z=static_z")

        # print("Generating", sampleSize * self.batch_size, "images for evaluation with GAM")

        selectedImages = np.zeros((sampleSize * self.batch_size, self.output_height, self.output_width, self.c_dim),
                                  dtype=np.float32)  # tf.stack(newBatch)
        # selectedNoise = np.zeros((self.batch_size, self.z_dim), dtype=np.float32) #tf.stack(newBatch)

        # if (dataset != None and improved_z_noise):
        #     print("Current GAN is using improved_z_noise")

        for b in range(0, sampleSize):

            eval_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

            if (static_z != None):
                eval_z = static_z[b * self.batch_size:(b + 1) * self.batch_size]

            if (dataset != None and improved_z_noise):
                batch_files = dataset[b * self.batch_size:(b + 1) * self.batch_size]
                basewidth = 10
                indexCounter = 0
                for imgName in batch_files:
                    if (cifarDataset):
                        # img = imgName
                        # print(imgName[0][0])

                        intData = (imgName + 1) * 128.
                        rawData = np.array(intData, np.uint8)
                        # print(intData[0][0])
                        # intData = imgName.reshape(256,256,3)
                        img = Image.fromarray(rawData)
                    else:
                        img = Image.open(imgName)
                    # img = img.convert('L')  # convert image to greyscale
                    wpercent = (basewidth / float(img.size[0]))
                    hsize = int((float(img.size[1]) * float(wpercent)))
                    img = img.resize((basewidth, basewidth), PIL.Image.ANTIALIAS)  # resizes the image to 10x10
                    img = img.convert('L')  # convert image to black and white
                    # name = 'processedImages/' + str(counter) + '.jpeg'
                    # img = ImageOps.invert(img)
                    # img.save(name)

                    pix = np.array(img, np.float32)
                    pix = (pix - 128) / 128  # Scales the pixels to be between -1 and 1
                    pix = pix.flatten()  # flattens the image to a single array of lenght of 100

                    eval_z[indexCounter] = pix  # Adds the image to the z-array
                    indexCounter += 1




            samples_eval = self.sess.run(
                [self.generatorOuput],
                feed_dict={
                    self.z: eval_z,
                }
            )

            images = np.asarray(samples_eval[0])

            for i in range(0, len(samples_eval)):
                # noe[0][i] = images[int(sortedIndeciesBatch[i])]
                selectedImages[b * self.batch_size + i] = images[i]

                # if(b == sampleSize/2):
                #     print(" - Halfway done")

        # print("Sample generation done")
        # print("Size: ", len(selectedImages))
        # print("Size[0]: ", len(selectedImages[0]))
        # print("Size[0][0]: ", len(selectedImages[0][0]))
        return selectedImages

    def batchGenerator(self, tournament_selection):

        if (tournament_selection == 1):
            getBestImagesOnly = True
            mixed = False
        elif (tournament_selection == 2):
            getBestImagesOnly = False
            mixed = False
        else:
            getBestImagesOnly = False
            mixed = True

        # eval_z = np.random.uniform(-1, 1, [64, self.z_dim]).astype(np.float32)
        # return self.generator(z, reuse=True)

        # eval_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        # samples_eval = self.sess.run(
        #     [self.G],
        #     feed_dict={
        #         self.z: eval_z,
        #     }
        # )
        #
        # return  samples_eval
        # print("Tournemetn selection in progress, generating from a size of ", self.evalSize)


        generationSize = self.batch_size * 5
        # images = self.generator(z,reuse=False, tempBatchSize=generationSize)
        scores = []
        # getBestImagesOnly = True
        indeices = list(range(0, self.evalSize))

        # eval_z = np.random.uniform(-1, 1, [64, self.z_dim]).astype(np.float32)
        print("self.evalSize",self.evalSize)
        eval_z = np.random.uniform(-1, 1, [self.evalSize, self.z_dim]).astype(np.float32)
        # eval_z = np.zeros((self.evalSize, self.z_dim, self.output_width, self.c_dim), dtype=np.float32)
        #

        # print(eval_z)
        #
        # self.G
        samples_eval = self.sess.run(
            [self.generatorOuput],
            feed_dict={
                self.z: eval_z,
            }
        )

        # return samples_eval

        # samples_eval = self.generator( eval_z, y=None,reuse=True, tempBatchSize=self.evalSize, useBatching=False)

        images = np.asarray(samples_eval[0])
        # print("samples_eval shape",samples_eval.shape)
        # print("Images shape",images.shape)
        #
        # print("!!!!!!!!!!!!!!!!!!!!!!!!! ")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!! ")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!! ")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!! ")
        # print("!!!!!!!!!!!!!!!!!!!!!!!!! ")

        # try:
        # hei = images[0]
        # except:
        #     return self.generator(eval_z,reuse=True)




        # try:
        batch_idxs = min(self.evalSize, np.inf) // self.batch_size
        # except:
        #     return samples_eval
        # print("!!!!!!!!!!!!!!!!!!!!!!!!! ", len(images))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!! batch_idxs:", (batch_idxs))
        # print(" ")

        counter = 0

        for idx in range(0, int(batch_idxs)):
            batch_images = images[idx * self.batch_size:(idx + 1) * self.batch_size]
            discriminatorScoresBatch = self.sess.run([self.discriminatorOutput],
                                                     feed_dict={self.inputs: batch_images})

            # scores.append(list(discriminatorScoresBatch[0][0]))


            for res in range(0, self.batch_size):
                scores.append(discriminatorScoresBatch[0][0][res][0])

        # print("Done. Size: ", len(samples_eval))
        # print("Done. Size[0]: ", len(samples_eval[0]))
        # print("Done. Size[0][0]: ", len(samples_eval[0][0]))
        # print("Done. Size[0][0][0]: ", len(samples_eval[0][0][0]))
        # print("Done. Size[0][0][0][0]: ", len(samples_eval[0][0][0][0]))


        # print("scores:",scores)
        # print("indeices:",indeices)

        sortedIndecies = [x for (y, x) in sorted(zip(scores, indeices), key=lambda pair: pair[0])]

        if (mixed):
            print("Sampling both good and bad noise")
            splitIndexFirstBatch = int(float(self.batch_size) / float(3))
            sortedIndeciesBatchWorst = sortedIndecies[:splitIndexFirstBatch]

            selectedImages = np.zeros((self.batch_size, self.output_height, self.output_width, self.c_dim),
                                      dtype=np.float32)  # tf.stack(newBatch)
            # selectedNoise = np.zeros((self.batch_size, self.z_dim), dtype=np.float32) #tf.stack(newBatch)
            selectedNoise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

            for i in range(0, splitIndexFirstBatch):
                # noe[0][i] = images[int(sortedIndeciesBatch[i])]
                selectedImages[i] = images[int(sortedIndeciesBatchWorst[i])]
                selectedNoise[i] = eval_z[int(sortedIndeciesBatchWorst[i])]

            sortedIndecies.reverse()
            sortedIndeciesBatchBest = sortedIndecies[:splitIndexFirstBatch]

            for i in range(0, splitIndexFirstBatch):
                # noe[0][i] = images[int(sortedIndeciesBatch[i])]
                selectedImages[i + splitIndexFirstBatch] = images[int(sortedIndeciesBatchBest[i])]
                selectedNoise[i + splitIndexFirstBatch] = eval_z[int(sortedIndeciesBatchBest[i])]

            return selectedImages, selectedNoise

        # print(sortedIndecies)
        elif (getBestImagesOnly):
            sortedIndecies.reverse()

        sortedIndeciesBatch = sortedIndecies[:self.batch_size]

        selectedImages = np.zeros((self.batch_size, self.output_height, self.output_width, self.c_dim),
                                  dtype=np.float32)  # tf.stack(newBatch)
        # selectedNoise = np.zeros((self.batch_size, self.z_dim), dtype=np.float32) #tf.stack(newBatch)
        selectedNoise = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        for i in range(0, self.batch_size):
            # noe[0][i] = images[int(sortedIndeciesBatch[i])]
            selectedImages[i] = images[int(sortedIndeciesBatch[i])]
            selectedNoise[i] = eval_z[int(sortedIndeciesBatch[i])]

        return selectedImages, selectedNoise

    def discriminator(self, image, y=None, reuse=False, tempBatchSize=None, secondDiscriminator=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if tempBatchSize == None:
                tempBatchSize = self.batch_size

            if not self.y_dim:
                if not secondDiscriminator:
                    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                    h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
                    h4 = linear(tf.reshape(h3, [tempBatchSize, -1]), 1, 'd_h3_lin')

                    return tf.nn.sigmoid(h4), h4
                else:
                    print("Hei hei hei")
                    h0 = lrelu(conv2d(image, self.df_dim, name='d2_h0_conv'))
                    h1 = lrelu(self.d2_bn1(conv2d(h0, self.df_dim * 2, name='d2_h1_conv')))
                    h2 = lrelu(self.d2_bn2(conv2d(h1, self.df_dim * 4, name='d2_h2_conv')))
                    h3 = lrelu(self.d2_bn3(conv2d(h2, self.df_dim * 8, name='d2_h3_conv')))
                    h4 = linear(tf.reshape(h3, [tempBatchSize, -1]), 1, 'd2_h3_lin')

                    return tf.nn.sigmoid(h4), h4
            else:
                yb = tf.reshape(y, [tempBatchSize, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [tempBatchSize, -1])
                h1 = concat([h1, y], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None, reuse=False, tempBatchSize=None, useBatching=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            if tempBatchSize == None:
                tempBatchSize = self.batch_size
            # elif(self.test and useBatching):
            #     tempBatchSize = self.evalSize


            # if(self.test):
            # print("askhd akdhakjs dkjahdjkahdkajshdksajdh kjashdjk adasd ")
            if not self.y_dim:

                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [tempBatchSize, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [tempBatchSize, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [tempBatchSize, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [tempBatchSize, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                # if(self.test and useBatching):
                #     print("Using batching!!!!!")
                # return self.batchGenerator(tf.nn.tanh(h4))

                # self.discriminator(tf.nn.tanh(h4), reuse=True)

                # Some tensor we want to print the value of


                if (useBatching and False):

                    return self.batchGenerator(tf.nn.tanh(h4))
                    # # return self.generator(z,reuse=True)

                    # try:
                    #     return self.batchGenerator()
                    # except:
                    #     print("Jeg feila!")
                    #     return tf.nn.tanh(h4)

                    # return tf.nn.tanh(h4)
                    # try:
                    #     test = self.batchGenerator()
                    #
                    #     a = tf.constant([1.0, 3.0])
                    #
                    #     # Add print operation
                    #     a = tf.Print(a, [a], message="Jeg greide det faktisk!")
                    #
                    #     # Add more elements of the graph using a
                    #     b = tf.add(a, a).eval()
                    #     return test
                    #
                    # except:
                    #
                    #     a = tf.constant([1.0, 3.0])
                    #
                    #     # Add print operation
                    #     a = tf.Print(a, [a], message="Jeg feila! ")
                    #
                    #     # Add more elements of the graph using a
                    #     b = tf.add(a, a).eval()
                    #
                    #     test = tf.nn.tanh(h4)
                    #     return test
                else:

                    # return np.ones((self.batch_size, self.output_height, self.output_width, self.c_dim), dtype=np.float32)

                    # return tf.stack(np.zeros((self.batch_size, self.output_height, self.output_width, self.c_dim), dtype=np.float32)) #tf.stack(newBatch)
                    return tf.nn.tanh(h4)
            else:

                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [tempBatchSize, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [tempBatchSize, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                    [tempBatchSize, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                    deconv2d(h2, [tempBatchSize, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def evaluators(self, z, y_eval=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y_eval, [self.evalSize, 1, 1, self.y_dim])
                z = concat([z, y_eval], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y_eval], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.evalSize, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.evalSize, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.evalSize, s_h, s_w, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        return X / 255., y_vec

    def load_mnist_with_test(self):
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY).astype(np.int)
        teY = np.asarray(teY).astype(np.int)

        # X = np.concatenate((trX, teX), axis=0)
        # y = np.concatenate((trY, teY), axis=0).astype(np.int)

        X = trX
        XT = teX

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(trY)

        np.random.seed(seed)
        np.random.shuffle(XT)
        np.random.seed(seed)
        np.random.shuffle(teY)

        y_vec = np.zeros((len(trY), self.y_dim), dtype=np.float)
        for i, label in enumerate(trY):
            y_vec[i, trY[i]] = 1.0

        y_vec_testing = np.zeros((len(teY), self.y_dim), dtype=np.float)
        for i, label in enumerate(teY):
            y_vec_testing[i, teY[i]] = 1.0

        return X / 255., y_vec, XT / 255., y_vec_testing

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def geCifar(self):
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

    def evalPastCheckpoints(self, checkpoint_dir, testSamples, generatedTestSamples, FLAGS, maxIterationNumber=None):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        print("Found", len(ckpt.all_model_checkpoint_paths), "checkpoints")
        averageScore = 0
        averageValidationScore = 0

        epochNumber = 0

        for checkpoint in ckpt.all_model_checkpoint_paths:
            ckpt_name = os.path.basename(checkpoint)
            iteratons = ckpt_name.split("-")[1]

            # self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # print(" [*] Success to read {}".format(ckpt_name))
            if (maxIterationNumber != None):

                if (int(iteratons) > maxIterationNumber):
                    print("This model is trained longer than the other. Aborting test here")
                    break

            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # print(" [*] Success to read {}".format(ckpt_name))
            score = self.evalImages(generatedTestSamples, FLAGS, False)
            validationScore = self.evalImages(testSamples, FLAGS, True)
            print("Epoch", epochNumber, "[" + str(iteratons) + " iterations]", " sample score is",
                  round((1 - score) * 100, 2), "% - (ValidationScore", round((1 - validationScore) * 100, 2), "%)")
            averageScore += score
            averageValidationScore += validationScore
            epochNumber += 1
        print(" ")
        print(" ")
        print(" ")
        averageScore = float(averageScore) / float(len(ckpt.all_model_checkpoint_paths))
        averageValidationScore = float(averageValidationScore) / float(len(ckpt.all_model_checkpoint_paths))
        print("Average score: ", averageScore)
        print("Average validation score: ", averageValidationScore)
        return averageScore, averageValidationScore

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # print("ckpt: ", ckpt)
        # print("ckpt: ", (ckpt.all_model_checkpoint_paths))

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    def loadSilent(self, checkpoint_dir):
        # print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # print(" [*] Success to read {}".format(ckpt_name))
            iteratons = ckpt_name.split("-")
            # print("     Trained for",iteratons,"iterations")
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    def loadTrainingITerations(self, checkpoint_dir):
        # print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # print(" [*] Success to read {}".format(ckpt_name))
            iteratons = ckpt_name.split("-")[1]
            return int(iteratons)

        else:
            print(" [*] Failed to find a checkpoint")
            return -1

    def getNumberOfCheckpoints(self, checkpoint_dir):
        checkpoint_dir_1 = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir_1)
        numberOfCheckpoints = ckpt.all_model_checkpoint_paths

        return len(numberOfCheckpoints)

    def loadCloestsCheckpoint(self, checkpoint_dir, numberOfITerations):

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        for checkpoint in ckpt.all_model_checkpoint_paths:
            ckpt_name = os.path.basename(checkpoint)

            if (numberOfITerations != None):
                iteratons = ckpt_name.split("-")[1]
                if (int(iteratons) == numberOfITerations):
                    # print("Found checkpoint match")
                    self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                    # print(" [*] Success to read {}".format(ckpt_name))
                    return True

        print("Mathcing checkpoint not found!")
        return False

    def loadCloestsCheckpointNumber(self, checkpoint_dir, numberOfITerations, checkpointNumber):

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        checkpoint = ckpt.all_model_checkpoint_paths[checkpointNumber]
        # for checkpoint in ckpt.all_model_checkpoint_paths:
        ckpt_name = os.path.basename(checkpoint)

        iteratons = ckpt_name.split("-")[1]
        if (int(iteratons) > numberOfITerations):
            # print("Stopping here, reached end")
            return True
        else:

            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # print("Mathcing checkpoint not found!")
            return False
