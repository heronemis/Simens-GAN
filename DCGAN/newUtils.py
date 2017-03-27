import os
import time
from datetime import datetime

sample_dir = 'samples'
currentSampleDir = ''
dataset = 'mnist'







def createFolderName(FLAGS):
    sample_dir = FLAGS.sample_dir
    dataset = FLAGS.dataset

    timeString = time.strftime("%d.%m.%Y")
    n = datetime.now()
    t = n.timetuple()
    y, m, d, h, min, sec, wd, yd, i = t
    if (min < 10):
        min = '0' + str(min)
    if (h < 10):
        h = '0' + str(h)

    folderName = dataset + " - " + timeString #+ " - " + (str(h) + "h" + str(min) + "m")
    print(folderName)


    addons = []
    if(FLAGS.improved_z_noise):
        addons.append("improved_z_noise")
    if(FLAGS.static_z):
        addons.append("static_z")
    if(FLAGS.minibatch_discrimination):
        addons.append("minibatch_discrimination")
    if(FLAGS.tournament_selection == 1):
        addons.append("tournament_selection_best")
    elif(FLAGS.tournament_selection == 2):
        addons.append("tournament_selection_worst")

    if(len(addons) > 0):
        folderName += " - " + str(addons).replace("'", "")

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    path = sample_dir + "/" + dataset
    if not os.path.exists(path):
        os.makedirs(path)

    currentFolder = sample_dir + "/" + dataset + "/" + folderName
    if os.path.exists(currentFolder):
        print("The folder allready excsists. Files will be overwritten")
    else:
        os.makedirs(currentFolder)

    FLAGS.sample_dir = currentFolder
    return FLAGS