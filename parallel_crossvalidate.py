# Reads all volumes meeting a given set of criteria,
# and uses a leave-one-out strategy to distinguish
# reviewed volumes (class 1) from random
# class 0. In cases where an author occurs more
# than once in the dataset, it leaves out all
# volumes by that author whenever making a prediction
# about one of them.

import numpy as np
import pandas as pd
import csv
import os
import sys
from collections import Counter
from multiprocessing import Pool
from models import LogisticRegression
import modelingprocess
import metafilter
# from scipy.stats import norm
import matplotlib.pyplot as plt

usedate = False
# Leave this flag false unless you plan major
# surgery to reactivate the currently-deprecated
# option to use "date" as a predictive feature.

# There are three different date types we can use.
# Choose which here.

# FUNCTIONS GET DEFINED BELOW.

def infer_date(metadictentry, datetype):
    if datetype == 'pubdate':
        return metadictentry[datetype]
    elif datetype == 'firstpub':
        firstpub = metadictentry['firstpub']
        if firstpub > 1700 and firstpub < 1950:
            return firstpub
        else:
            return metadictentry['pubdate']
    else:
        sys.exit(0)

def appendif(key, value, dictionary):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]

# Clean this up and make it unnecessary.

def dirty_pairtree(htid):
    ''' Changes a 'clean' HathiTrust ID (with only chars that are
    legal in filenames) into a 'clean' version of the same name
    (which may contain illegal chars.)
    '''
    period = htid.find('.')
    prefix = htid[0:period]
    postfix = htid[(period+1):]
    if '=' in postfix:
        postfix = postfix.replace('+', ':')
        postfix = postfix.replace('=', '/')
    dirtyname = prefix + "." + postfix
    return dirtyname

def forceint(astring):
    try:
        intval = int(astring)
    except:
        intval = 0

    return intval

def get_features(wordcounts, wordlist):
    numwords = len(wordlist)
    wordvec = np.zeros(numwords)
    for idx, word in enumerate(wordlist):
        if word in wordcounts:
            wordvec[idx] = wordcounts[word]

    return wordvec

# In an earlier version of this script, we sometimes used
# "publication date" as a feature, to see what would happen.
# In the current version, we don't. Some of the functions
# and features remain, but they are deprecated. E.g.:

def get_features_with_date(wordcounts, wordlist, date, totalcount):
    numwords = len(wordlist)
    wordvec = np.zeros(numwords + 1)
    for idx, word in enumerate(wordlist):
        if word in wordcounts:
            wordvec[idx] = wordcounts[word]

    wordvec = wordvec / (totalcount + 0.0001)
    wordvec[numwords] = date
    return wordvec

def sliceframe(dataframe, yvals, excludedrows, testrow):
    newyvals = list(yvals)
    for i in excludedrows:
        del newyvals[i]
        # NB: This only works if we assume that excluded rows
        # has already been sorted in descending order !!!!!!!

    trainingset = dataframe.drop(dataframe.index[excludedrows])

    newyvals = np.array(newyvals)
    testset = dataframe.iloc[testrow]

    return trainingset, newyvals, testset

def normalizearray(featurearray, usedate):
    '''Normalizes an array by centering on means and
    scaling by standard deviations. Also returns the
    means and standard deviations for features, so that
    they can be pickled.
    '''

    numinstances, numfeatures = featurearray.shape
    means = list()
    stdevs = list()
    lastcolumn = numfeatures - 1
    for featureidx in range(numfeatures):

        thiscolumn = featurearray.iloc[:, featureidx]
        thismean = np.mean(thiscolumn)

        thisstdev = np.std(thiscolumn)

        if (not usedate) or featureidx != lastcolumn:
            # If we're using date we don't normalize the last column.
            means.append(thismean)
            stdevs.append(thisstdev)
            featurearray.iloc[:, featureidx] = (thiscolumn - thismean) / thisstdev
        else:
            print('FLAG')
            means.append(thismean)
            thisstdev = 0.1
            stdevs.append(thisstdev)
            featurearray.iloc[:, featureidx] = (thiscolumn - thismean) / thisstdev
            # We set a small stdev for date.

    return featurearray, means, stdevs

# Deprecated; see note below on feature selection
# def binormal_select(vocablist, positivecounts, negativecounts, totalpos, totalneg, k):
#     ''' A feature-selection option, not currently in use.
#     '''
#     all_scores = np.zeros(len(vocablist))
#
#     for idx, word in enumerate(vocablist):
#         # For each word we create a vector the length of vols in each class
#         # that contains real counts, plus zeroes for all those vols not
#         # represented.
#
#         positives = np.zeros(totalpos, dtype='int64')
#         if word in positivecounts:
#             positives[0: len(positivecounts[word])] = positivecounts[word]
#
#         negatives = np.zeros(totalneg, dtype='int64')
#         if word in negativecounts:
#             negatives[0: len(negativecounts[word])] = negativecounts[word]
#
#         featuremean = np.mean(np.append(positives, negatives))
#
#         tp = sum(positives > featuremean)
#         fp = sum(positives <= featuremean)
#         tn = sum(negatives > featuremean)
#         fn = sum(negatives <= featuremean)
#         tpr = tp/(tp+fn)  # true positive ratio
#         fpr = fp/(fp+tn)  # false positive ratio
#
#         bns_score = abs(norm.ppf(tpr) - norm.ppf(fpr))
#         # See Forman
#
#         if np.isinf(bns_score) or np.isnan(bns_score):
#             bns_score = 0
#
#         all_scores[idx] = bns_score
#
#     zipped = [x for x in zip(all_scores, vocablist)]
#     zipped.sort(reverse=True)
#     with open('bnsscores.tsv', mode='w', encoding='utf-8') as f:
#         for score, word in zipped:
#             f.write(word + '\t' + str(score) + '\n')
#
#     return [x[1] for x in zipped[0:k]]

def fetch_volume_paths(sourcefolder):
    if not sourcefolder.endswith('/'):
        sourcefolder = sourcefolder + '/'

    # This just makes things easier.

    # Get a list of files.
    allthefiles = os.listdir(sourcefolder)
    # random.shuffle(allthefiles)

    volumeIDs = list()
    volumepaths = list()

    for filename in allthefiles:

        if filename.endswith(extension):
            volID = filename.replace(extension, "")
            # The volume ID is basically the filename minus its extension.
            # Extensions are likely to be long enough that there is little
            # danger of accidental occurrence inside a filename. E.g.
            # '.fic.tsv'
            path = sourcefolder + filename
            volumeIDs.append(volID)
            volumepaths.append(path)
    return volumepaths, volumeIDs

def create_model(paths, exclusions, thresholds, classifyconditions):
    ''' This is the main function in the module.
    It can be called externally; it's also called
    if the module is run directly.
    '''

    sourcefolder, extension, classpath, outputpath = paths
    excludeif, excludeifnot, excludebelow, excludeabove, sizecap = exclusions
    pastthreshold, futurethreshold = thresholds
    category2sorton, positive_class, datetype, numfeatures, regularization, penalty = classifyconditions

    verbose = False

    volumeIDs, volumepaths = fetch_volume_paths(sourcefolder)

    metadict = metafilter.get_metadata(classpath, volumeIDs, excludeif, excludeifnot, excludebelow, excludeabove)

    # Now that we have a list of volumes with metadata, we can select the groups of IDs
    # that we actually intend to contrast. If we want to us more or less everything,
    # this may not be necessary. But in some cases we want to use randomly sampled subsets.

    # The default condition here is

    # category2sorton = 'reviewed'
    # positive_class = 'rev'
    # sizecap = 350
    # A sizecap less than one means, no sizecap.

    IDsToUse, classdictionary = metafilter.label_classes(metadict, category2sorton, positive_class, sizecap)

    # make a vocabulary list and a volsize dict
    wordcounts = Counter()

    volspresent = list()
    orderedIDs = list()

    # Required for feature selection, deprecated
    # positivecounts = dict()
    # negativecounts = dict()

    for volid, volpath in zip(volumeIDs, volumepaths):
        if volid not in IDsToUse:
            continue
        else:
            volspresent.append((volid, volpath))
            orderedIDs.append(volid)

        date = infer_date(metadict[volid], datetype)
        if date < pastthreshold or date > futurethreshold:
            continue
        else:
            with open(volpath, encoding='utf-8') as f:
                for line in f:
                    fields = line.strip().split('\t')
                    if len(fields) > 2 or len(fields) < 2:
                        # print(line)
                        continue
                    word = fields[0]
                    if len(word) > 0 and word[0].isalpha():
                        count = int(fields[1])
                        wordcounts[word] += 1
                        # for initial feature selection we use the number of
                        # *documents* that contain a given word,
                        # so it's just +=1.

    vocablist = [x[0] for x in wordcounts.most_common(numfeatures)]
    # vocablist = [
    #     'eyes', 'mission', 'hurrying', 'whereon', 'hollow', 'sake', 'sign',
    #     'wondrous', 'greet', 'spotless', 'brows', 'trance', 'harsh', 'black',
    #     "i'll", 'wooed', 'command', 'haired', 'stir', 'slow', 'smooth',
    #     'ghosts', 'anxious', 'appear', 'shuddering', 'reign', 'guide', 'half',
    #     'hers', 'follows', 'dashed', 'throng', 'oft', 'scorn', 'evil',
    #     'change', 'tis', 'moments', 'cheer', 'wert', 'what', 'goal', 'raging',
    #     'blew', 'hurt', 'waved', 'loving', 'thorn', 'earthly', 'western',
    #     'fainting', 'goes', 'reigns', 'grass', 'shut', 'turf', 'nights',
    #     'fresh', 'branch', 'dusk', 'sank', 'rays', 'steal', 'strand', 'wrote',
    #     'toiled', 'purest', 'fondly', 'sighed', 'true', 'sealed', 'heavy',
    #     'plan', 'steeds', 'ills', 'turning', 'appeared', 'helpless', 'dull',
    #     'mock', 'drunk', 'visions', 'enough', 'sheen', 'moan', 'lately',
    #     'wheel', 'parted', 'unheard', 'not', 'wherein', 'nation', 'trembled',
    #     "we'll", 'folly', 'worn', 'our', 'lest', 'gulf', 'magic', 'brave',
    #     'whirl', 'round', 'weal', 'wind', 'grand', 'raised', 'red',
    #     'anything', 'until', 'wound', 'shook', 'saviour', 'daughters', 'once',
    #     'shattered', 'band', 'whose', 'bold', 'nought', 'winning', 'has',
    #     'sees', 'blowing', 'yours', 'inward', 'blown', 'roaring', 'sped',
    #     'petals', 'pay', 'meek', 'clear', 'sand', 'pomp', 'graves', 'thank',
    #     'lords', 'rose', 'sacred', 'shakes', 'springtime', 'joyful', 'flesh',
    #     'nature', 'violets', 'ghost', 'fixed', 'search', 'fire', 'startled',
    #     'twelve', 'and', 'stain', 'freed', 'darkly', 'breath', 'seasons',
    #     'conquer', 'still', 'tell', 'brooding', 'last', 'cold', 'doom', 'seen',
    #     'keeps', 'shadow', 're', 'plant', 'presence', 'four', 'own', 'thus',
    #     'every', 'crime', 'more', 'sweets', 'fancy', 'often', 'rich', 'fair',
    #     'lord'
    # ]

    # vocablist = binormal_select(vocablist, positivecounts, negativecounts, totalposvols, totalnegvols, 3000)
    # Feature selection is deprecated. There are cool things
    # we could do with feature selection,
    # but they'd improve accuracy by 1% at the cost of complicating our explanatory task.
    # The tradeoff isn't worth it. Explanation is more important.
    # So we just take the most common words (by number of documents containing them)
    # in the whole corpus. Technically, I suppose, we could crossvalidate that as well,
    # but *eyeroll*.
    # To enable feature selection, uncomment binormal_select above too

    donttrainon = list()

    # Here we create a list of volumed IDs not to be used for training.
    # For instance, we have supplemented the dataset with volumes that
    # are in the Norton but that did not actually occur in random
    # sampling. We want to make predictions for these, but never use
    # them for training.

    for idx1, anid in enumerate(orderedIDs):
        reviewedstatus = metadict[anid]['reviewed']
        date = infer_date(metadict[anid], datetype)
        if reviewedstatus == 'addedbecausecanon':
            donttrainon.append(idx1)
        elif date < pastthreshold or date > futurethreshold:
            donttrainon.append(idx1)

    authormatches = [list(donttrainon) for x in range(len(orderedIDs))]
    # For every index in authormatches, identify a set of indexes that have
    # the same author. Obvs, there will always be at least one.

    # Since we are going to use these indexes to exclude rows, we also add
    # all the ids in donttrainon to every volume

    for idx1, anid in enumerate(orderedIDs):
        thisauthor = metadict[anid]['author']
        for idx2, anotherid in enumerate(orderedIDs):
            otherauthor = metadict[anotherid]['author']
            if thisauthor == otherauthor and idx2 not in authormatches[idx1]:
                authormatches[idx1].append(idx2)

    for alist in authormatches:
        alist.sort(reverse=True)

    # I am reversing the order of indexes so that I can delete them from
    # back to front, without changing indexes yet to be deleted.
    # This will become important in the modelingprocess module.

    volsizes = dict()
    voldata = list()
    classvector = list()

    for volid, volpath in volspresent:

        with open(volpath, encoding='utf-8') as f:
            voldict = dict()
            totalcount = 0
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) > 2 or len(fields) < 2:
                    continue

                word = fields[0]
                count = int(fields[1])
                voldict[word] = count
                totalcount += count

        date = infer_date(metadict[volid], datetype)
        date = date - 1700
        if date < 0:
            date = 0

        if usedate:
            features = get_features_with_date(voldict, vocablist, date, totalcount)
            voldata.append(features)
        else:
            features = get_features(voldict, vocablist)
            voldata.append(features / (totalcount + 0.001))

        volsizes[volid] = totalcount
        classflag = classdictionary[volid]
        classvector.append(classflag)

    data = pd.DataFrame(voldata)

    model_args = list()
    for i, volid in enumerate(orderedIDs):
        listtoexclude = authormatches[i]
        arg_set = data, classvector, listtoexclude, i, usedate, regularization, penalty
        model_args.append(arg_set)

    # Now do leave-one-out predictions.
    print('Beginning multiprocessing.')

    pool = Pool(processes=4)
    res = pool.map_async(modelingprocess.model_one_volume, model_args)

    # After all files are processed, write metadata, errorlog, and counts of phrases.
    res.wait()
    resultlist = res.get()

    assert len(resultlist) == len(orderedIDs)

    logisticpredictions = dict()
    for i, volid in enumerate(orderedIDs):
        logisticpredictions[volid] = resultlist[i]

    pool.close()
    pool.join()

    print('Multiprocessing concluded.')

    truepositives = 0
    truenegatives = 0
    falsepositives = 0
    falsenegatives = 0
    allvolumes = list()

    with open(outputpath, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['volid', 'reviewed', 'obscure', 'pubdate', 'birthdate', 'gender', 'nation', 'allwords', 'logistic', 'author', 'title', 'pubname', 'actually', 'realclass']
        writer.writerow(header)
        for volid in IDsToUse:
            metadata = metadict[volid]
            reviewed = metadata['reviewed']
            obscure = metadata['obscure']
            pubdate = infer_date(metadata, datetype)
            birthdate = metadata['birthdate']
            gender = metadata['gender']
            nation = metadata['nation']
            author = metadata['author']
            title = metadata['title']
            canonicity = metadata['canonicity']
            pubname = metadata['pubname']
            allwords = volsizes[volid]
            logistic = logisticpredictions[volid]
            realclass = classdictionary[volid]
            outrow = [volid, reviewed, obscure, pubdate, birthdate, gender, nation, allwords, logistic, author, title, pubname, canonicity, realclass]
            writer.writerow(outrow)
            allvolumes.append(outrow)

            if logistic > 0.5 and classdictionary[volid] > 0.5:
                truepositives += 1
            elif logistic <= 0.5 and classdictionary[volid] < 0.5:
                truenegatives += 1
            elif logistic <= 0.5 and classdictionary[volid] > 0.5:
                falsenegatives += 1
            elif logistic > 0.5 and classdictionary[volid] < 0.5:
                falsepositives += 1

    donttrainon.sort(reverse=True)
    trainingset, yvals, testset = sliceframe(data, classvector, donttrainon, 0)
    newmodel = LogisticRegression(C=regularization, penalty=penalty)
    trainingset, means, stdevs = normalizearray(trainingset, usedate)
    print(trainingset.shape)
    print(yvals.shape)
    newmodel.fit(trainingset, yvals)

    coefficients = newmodel.coef_[0] * 100

    coefficientuples = list(zip(coefficients, (coefficients / np.array(stdevs)), vocablist + ['pub.date']))
    coefficientuples.sort()
    if verbose:
        for coefficient, normalizedcoef, word in coefficientuples:
            print(word + " :  " + str(coefficient))

    print()
    accuracy = (truepositives + truenegatives) / len(IDsToUse)

    coefficientpath = outputpath.replace('.csv', '.coefs.csv')
    with open(coefficientpath, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for triple in coefficientuples:
            coef, normalizedcoef, word = triple
            writer.writerow([word, coef, normalizedcoef])

    return accuracy, allvolumes, coefficientuples

def diachronic_tilt(allvolumes, modeltype, datelimits):
    ''' Takes a set of predictions produced by a model that knows nothing about date,
    and divides it along a line with a diachronic tilt. We need to do this in a way
    that doesn't violate crossvalidation. I.e., we shouldn't "know" anything
    that the model didn't know. We tried a couple of different ways to do this, but
    the simplest and actually most reliable is to divide the whole dataset along a
    linear central trend line for the data!
    '''

    listofrows = list()
    classvector = list()

    # DEPRECATED
    # if modeltype == 'logistic' and len(datelimits) == 2:
    #     # In this case we construct a subset of data to model on.
    #     tomodeldata = list()
    #     tomodelclasses = list()
    #     pastthreshold, futurethreshold = datelimits

    for volume in allvolumes:
        date = volume[3]
        logistic = volume[8]
        realclass = volume[13]
        listofrows.append([logistic, date])
        classvector.append(realclass)

        # DEPRECATED
        # if modeltype == 'logistic' and len(datelimits) == 2:
        #     if date >= pastthreshold and date <= futurethreshold:
        #         tomodeldata.append([logistic, date])
        #         tomodelclasses.append(realclass)

    y, x = [a for a in zip(*listofrows)]
    plt.axis([min(x) - 2, max(x) + 2, min(y) - 0.02, max(y) + 0.02])
    reviewedx = list()
    reviewedy = list()
    randomx = list()
    randomy = list()

    for idx, reviewcode in enumerate(classvector):
        if reviewcode == 1:
            reviewedx.append(x[idx])
            reviewedy.append(y[idx])
        else:
            randomx.append(x[idx])
            randomy.append(y[idx])

    plt.plot(reviewedx, reviewedy, 'ro')
    plt.plot(randomx, randomy, 'k+')

    if modeltype == 'logistic':
        # all this is DEPRECATED
        print("Hey, you're attempting to use the logistic-tilt option")
        print("that we deactivated. Go in and uncomment the code.")

        # if len(datelimits) == 2:
        #     data = pd.DataFrame(tomodeldata)
        #     responsevariable = tomodelclasses
        # else:
        #     data = pd.DataFrame(listofrows)
        #     responsevariable = classvector

        # newmodel = LogisticRegression(C = 100000)
        # newmodel.fit(data, responsevariable)
        # coefficients = newmodel.coef_[0]

        # intercept = newmodel.intercept_[0] / (-coefficients[0])
        # slope = coefficients[1] / (-coefficients[0])

        # p = np.poly1d([slope, intercept])

    elif modeltype == 'linear':
        # what we actually do

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        slope = z[0]
        intercept = z[1]

    plt.plot(x, p(x), "b-")
    plt.show(block=False)

    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')
    classvector = np.array(classvector)
    dividingline = intercept + (x * slope)
    predicted_as_reviewed = (y > dividingline)
    really_reviewed = (classvector == 1)

    accuracy = sum(predicted_as_reviewed == really_reviewed) / len(classvector)

    return accuracy

if __name__ == '__main__':

    # If this class is called directly, it creates a single model using the default
    # settings set below.

    # PATHS.

    # sourcefolder = '/Users/tunder/Dropbox/GenreProject/python/reception/fiction/texts/'
    # extension = '.fic.tsv'
    # classpath = '/Users/tunder/Dropbox/GenreProject/python/reception/fiction/masterficmeta.csv'
    # outputpath = '/Users/tunder/Dropbox/GenreProject/python/reception/fiction/predictions.csv'

    sourcefolder = 'poems/'
    extension = '.poe.tsv'
    classpath = 'poemeta.csv'
    outputpath = 'logisticpredictions.csv'

    # We can simply exclude volumes from consideration on the basis on any
    # metadata category we want, using the dictionaries defined below.

    # EXCLUSIONS.

    excludeif = dict()
    excludeif['pubname'] = 'TEM'
    # We're not using reviews from Tait's.

    excludeif['recept'] = 'addcanon'
    # We don't ordinarily include canonical volumes that were not in either sample.
    # These are included only if we're testing the canon specifically.

    excludeifnot = dict()
    excludeabove = dict()
    excludebelow = dict()

    excludebelow['inferreddate'] = 1800
    excludeabove['inferreddate'] = 1950
    sizecap = 360

    # For more historically-interesting kinds of questions, we can limit the part
    # of the dataset that gets TRAINED on, while permitting the whole dataset to
    # be PREDICTED. (Note that we always exclude authors from their own training
    # set; this is in addition to that.) The variables futurethreshold and
    # pastthreshold set the chronological limits of the training set, inclusive
    # of the threshold itself.

    # THRESHOLDS

    pastthreshold = -1
    futurethreshold = 2000

    # CLASSIFY CONDITIONS

    positive_class = 'rev'
    category2sorton = 'reviewed'
    datetype = 'firstpub'
    numfeatures = 3200
    regularization = .00007

    paths = (sourcefolder, extension, classpath, outputpath)
    exclusions = (excludeif, excludeifnot, excludebelow, excludeabove, sizecap)
    thresholds = (pastthreshold, futurethreshold)
    classifyconditions = (category2sorton, positive_class, datetype, numfeatures, regularization)

    rawaccuracy, allvolumes, coefficientuples = create_model(paths, exclusions, thresholds, classifyconditions)

    print('If we divide the dataset with a horizontal line at 0.5, accuracy is: ', str(rawaccuracy))
    tiltaccuracy = diachronic_tilt(allvolumes, 'linear', [])

    print("Divided with a line fit to the data trend, it's ", str(tiltaccuracy))
