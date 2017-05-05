import os
import time
from datetime import datetime
import csv
import random

sample_dir = 'samples'
currentSampleDir = ''
dataset = 'mnist'

commonName = ""

combinedAccuarcy = []
n = 5


def localize_floats(row):
    return [
        str(el).replace('.', ',') if isinstance(el, float) else el
        for el in row
    ]


def initCSV(sampleDir):
    global commonName
    with open(sampleDir+"/accuracy - " + commonName + ".csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL, dialect='excel-tab')
        writer.writerow(["Real classified as real(%)","Fake classified as fake(%)","Combined","Avg combined(last " + str(n) + ")","Max combined(last " + str(n) + ")"])


def createConfingCSV(sampleDir,FLAGS):

    with open(sampleDir + "/config - " + commonName + ".csv", "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL, dialect='excel-tab')
        writer.writerow(["Setting", "Value"])

        for c in FLAGS.__flags:
            # print(c, "-",FLAGS.__flags[c])
            writer.writerow([c, FLAGS.__flags[c]])


def addStats(accuracy):
    global combinedAccuarcy
    combinedAccuarcy.append(accuracy[2])
    print("Current Accuarcy:",accuracy[2]*100,"%" )


    lastNRestuls = (combinedAccuarcy[-n:])
    avgScore = sum(lastNRestuls) / float(len(lastNRestuls))
    maxScore = max(lastNRestuls)

    accuracy.append(avgScore)
    accuracy.append(maxScore)

    return accuracy



def writeAccuracyToFile(sampleDir,accuracy):
    # mylist = [0.9989999,0.324234]
    #
    # with open('test.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(mylist)




    # RESULTS = [
    #     [50, 3, 1]
    # ]
    # resultFile = open("testExcel.csv", 'wb')
    # resultWriter = csv.writer(resultFile, dialect='excel-tab')
    # resultWriter.writerows(RESULTS)

    # with open('text_to_csv.csv', "w") as csvfile:
    #     writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL, dialect='excel-tab')
    #     writer.writerow(["Real(%)","Fake(%)"])

    updatedAcc = addStats(accuracy)

    with open(sampleDir+"/accuracy - " + commonName + ".csv", 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_MINIMAL, dialect='excel-tab')
        # for i in range(0, len(fList), 2):
        # mylist = [  random.uniform(0.000000, 1.0000000), random.uniform(0.00000000, 1.000000)  ]
        writer.writerow(localize_floats(updatedAcc))

    return updatedAcc[3]

    # with open('test.csv', "w") as output:
    #     writer = csv.writer(output, lineterminator='\n', delimiter=",")
    #     writer.writerow(["Real(%)","Fake(%)"])
    #
    #
    # with open(r'test.csv', 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(mylist)
    #
    #
    # with open(r'test.csv', 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(mylist)



def getAddons(FLAGS):
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

    if (len(addons) > 0):
        return "-" + str(addons).replace("'", "").replace(",", "+").replace("]", "").replace("[", "").replace(" ", "")
    else:
        return ""


def createFolderName(FLAGS):
    global commonName
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

    # if(len(addons) > 0):
    addonString = getAddons(FLAGS)
    folderName += addonString

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
    commonName = folderName
    FLAGS.checkpoint_dir = FLAGS.checkpoint_dir + "/" + dataset + addonString
    return FLAGS



# hei = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
# print(hei[-5:])


# writeAccuracyToFile()