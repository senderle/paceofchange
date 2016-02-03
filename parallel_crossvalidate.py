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
import random
import math
from collections import Counter, defaultdict
from multiprocessing import Pool
from models import LogisticRegression
import modelingprocess
import metafilter

import matplotlib
matplotlib.use("Agg")  # Get rid of the annoying Python rocket.
import matplotlib.pyplot as plt

def infer_date(metadictentry, datetype):
    if datetype == 'pubdate':
        return metadictentry[datetype]
    elif datetype == 'firstpub':
        firstpub = metadictentry['firstpub']
        if 1700 < firstpub < 1950:
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

class VolumeMeta(object):
    """A class representing a set of HathiTrust volumes with
    accompanying metadata. All attributes should be treated as
    read-only.
    """
    def __init__(self, sourcefolder, extension, classpath, exclusions,
                 category2sorton, positive_class):

        files = [f for f in os.listdir(sourcefolder) if f.endswith(extension)]
        all_ids = [f.rpartition(extension)[0] for f in files]

        exif, exifnot, exbelow, exabove, sizecap = exclusions
        self.meta = metafilter.get_metadata(
            classpath, all_ids, exif, exifnot, exbelow, exabove
        )
        self.label = metafilter.classlabels(
            self.meta, category2sorton, positive_class, sizecap
        )
        self.path = {vid: os.path.join(sourcefolder, filename)
                     for vid, filename in zip(all_ids, files)}

        self.ordered_ids = [vid for vid in all_ids
                            if vid in self.label]
        self._groupcache = {}

    def __iter__(self):
        return iter(self.ordered_ids)

    def __len__(self):
        return len(self.ordered_ids)

    def __getitem__(self, index):
        return self.ordered_ids[index]

    def infer_date(self, volid, datetype):
        return infer_date(self.meta[volid], datetype)

    def itermeta(self, key):
        if key is None:
            return (self.meta[oid] for oid in self)
        elif key == 'path':
            return (self.path[oid] for oid in self)
        elif key == 'label':
            return (self.label[oid] for oid in self)
        else:
            return (self.meta[oid][key] for oid in self)

    def zipmeta(self, *keys):
        if not keys:
            keys = [None]
        iters = [iter(self)] + [self.itermeta(k) for k in keys]
        return zip(*iters)

    def groupmeta(self, key):
        if key not in self._groupcache:
            groups = list(self.itermeta(key))
            group_indices = defaultdict(list)
            for i, group in enumerate(groups):
                group_indices[group].append(i)
            self._groupcache[key] = [group_indices[group] for group in groups]

        return self._groupcache[key]

class TrainingData(object):
    """A class that abstracts away the details of managing feature
    sets and training data. All attributes should be treated as
    read-only. Public methods are provided for holding out "test"
    data (for k-fold cross validation) and for changing feature
    sets (i.e. via vocablist).

    Public methods: ``next_testdata``, ``set_vocablist``.
    """
    def __init__(self, volumes, pastthreshold, futurethreshold,
                 datetype, numfeatures, vocab=None, test_step=None,
                 usedate=False):
        """NOTE: Leave ``usedate`` false unless you plan major surgery
        to reactivate the currently-deprecated option to use "date"
        as a predictive feature.
        """
        self.volumes = volumes
        self.pastthreshold = pastthreshold
        self.futurethreshold = futurethreshold
        self.datetype = datetype
        self.usedate = usedate

        self.test_start = self.test_end = 0
        self.test_step = test_step
        self.shuffled_ids, self.shuffled_indices = self.shuffled_testdata()
        self.dont_train = self._dont_train()
        self.authormatches = self._authormatches()

        self.numfeatures = numfeatures
        self.wordcounts = self.word_doc_counts()
        self.set_vocablist(vocab)
        self.voldata, self.volsizes = self._voldata()

    def volume_words(self, volid):
        """Iterate over the word, count pairs in a given volume's file."""
        volpath = self.volumes.path[volid]
        with open(volpath, encoding='utf-8') as f:
            rows = (l.strip().split('\t') for l in f)
            pairs = (row for row in rows if len(row) == 2)
            return [(word, int(count)) for word, count in pairs]

    def word_doc_counts(self):
        wordcounts = Counter()
        for volid in self.volumes:
            date = self.volumes.infer_date(volid, self.datetype)
            if self.pastthreshold <= date <= self.futurethreshold:
                # Ignore the counts for now; for feature selection, we
                # just want the number of documents that contain the word.
                features = (word for word, _count_unused
                            in self.volume_words(volid)
                            if len(word) > 0 and word[0].isalpha())
                wordcounts.update(features)
        return wordcounts

    def shuffled_testdata(self):
        """Create a random permutation of all training data that preserves
        the existing balance between the classes. To get random test and
        validation sets for simple validation, and testing, just slice out
        subsets. To do k-fold validation, iterate over slices of size k,
        training on the remainder. This makes no effort to balance over
        other categories.
        """
        # At some point, consider balancing over the cartesian product of
        # multiple categories (e.g. [labels] x [decades]). But that may
        # leave stragglers.
        label_to_id = defaultdict(list)
        for volid, label in self.volumes.zipmeta('label'):
            label_to_id[label].append(volid)

        # Randomize data within each class:
        for v in label_to_id.values():
            random.shuffle(v)

        # Interleave data from each class so that every slice is
        # approximately balanced among all classes. This assumes the
        # classes are already balanced to begin with!
        interleaved = zip(*label_to_id.values())
        shuf_ids = [volid for idtuple in interleaved for volid in idtuple]

        id_to_ordered = {volid: i for i, volid in enumerate(self.volumes)}
        shuf_indices = [id_to_ordered[tid] for tid in shuf_ids]
        return shuf_ids, shuf_indices

    def next_testdata(self):
        if self.test_step is None:
            return False

        self.test_end += self.test_step
        # Because test_start and test_end both start at 0...
        self.test_start = self.test_end - self.test_step
        self.dont_train = self._dont_train()
        self.authormatches = self._authormatches()

        # Tell caller whether any test data remains.
        return bool(self.test_indices)

    @property
    def test_indices(self):
        return self.shuffled_indices[self.test_start:self.test_end]

    def _dont_train(self):
        dont_train = list()

        # Here we create a list of volume IDs not to be used for training.
        # For instance, we have supplemented the dataset with volumes that
        # are in the Norton but that did not actually occur in random
        # sampling. We want to make predictions for these, but never use
        # them for training.
        testset = set(self.test_indices)
        authorindex = self.volumes.groupmeta('author')
        for i, (vid, revstatus) in enumerate(self.volumes.zipmeta('reviewed')):
            date = self.volumes.infer_date(vid, self.datetype)
            if i in testset:
                dont_train.append(i)
                dont_train.extend(authorindex[i])
            elif revstatus == 'addedbecausecanon':
                dont_train.append(i)
            elif not self.pastthreshold <= date <= self.futurethreshold:
                dont_train.append(i)

        return dont_train

    def _authormatches(self):
        # Since we are going to use these indexes to exclude rows,
        # we also add all the ids in dont_train to every volume.
        combined = (self.dont_train + ai
                    for ai in self.volumes.groupmeta('author'))
        return [sorted(set(c)) for c in combined]

    def set_vocablist(self, vocab=None):
        if vocab is None:
            self.vocablist = [x[0] for x in
                              self.wordcounts.most_common(self.numfeatures)]
        else:
            self.vocablist = vocab
        self.voldata, self.volsizes = self._voldata()

    def _voldata(self):
        vdata = []
        vsizes = {}
        for volid in self.volumes:
            voldict = dict(self.volume_words(volid))
            totalcount = sum(v for k, v in voldict.items())
            date = self.volumes.infer_date(volid, self.datetype) - 1700
            date = 0 if date < 0 else date

            if self.usedate:
                features = get_features_with_date(voldict, self.vocablist,
                                                  date, totalcount)
            else:
                features = get_features(voldict, self.vocablist)
                features /= totalcount + 0.001
            vsizes[volid] = totalcount
            vdata.append(features)
        return vdata, vsizes

    @property
    def classvector(self):
        return list(self.volumes.itermeta('label'))

def write_coefficients(coefficientuples, outputpath):
    coefficientpath = outputpath.replace('.csv', '.coefs.csv')
    with open(coefficientpath, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for coef, normalizedcoef, word in coefficientuples:
            writer.writerow([word, coef, normalizedcoef])

def write_output_rows(header, allvolumes, outputpath):
    with open(outputpath, mode='w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in allvolumes:
            writer.writerow(row)

def read_word_coefs(words, filename):
    wordset = set(words)
    word_coefs = {}
    with open(filename) as f:
        rows = (line.split(',') for line in f)
        rows = [row for row in rows if len(row) == 3]
        for word, normed, coef in rows:
            if word in wordset:
                word_coefs[word] = coef
    return word_coefs

class _ValidationModel(object):
    def __init__(self, training, penalty, regularization, verbose=False):
        self.training = training
        self.data = pd.DataFrame(training.voldata)
        self.regularization = regularization
        self.penalty = penalty
        self.verbose = verbose
        self.predictions = self._predict()
        self.allvolumes = self._make_output_rows()
        self.coefficientuples = self._full_coefficients()

    def accuracy(self, predictions=None):
        if predictions is None:
            predictions = self.predictions
        truepositives = truenegatives = 0  # falsepositives = falsenegatives = 0
        n_predictions = 0
        for volid in self.training.volumes:
            # When testing, predictions will not necessarily be given
            # to all ids.
            if volid not in predictions:
                continue

            n_predictions += 1
            predict = predictions[volid] > 0.5
            actual = self.training.volumes.label[volid] > 0.5
            truepositives += int(predict and actual)
            truenegatives += int(not predict and not actual)
            # falsepositives += predict and not actual
            # falsenegatives += not predict and actual

        return (truepositives + truenegatives) / n_predictions

    def display_coefficients(self):
        for coefficient, normalizedcoef, word in self.coefficientuples:
            print(word + " :  " + str(coefficient))
        print()

    @property
    def header(self):
        return ['volid', 'reviewed', 'obscure', 'pubdate', 'birthdate',
                'gender', 'nation', 'allwords', 'logistic', 'author',
                'title', 'pubname', 'actually', 'realclass']

    def _make_output_rows(self):
        allvolumes = list()
        for volid, metadata in self.training.volumes.zipmeta():
            reviewed = metadata['reviewed']
            obscure = metadata['obscure']
            pubdate = infer_date(metadata, self.training.datetype)
            birthdate = metadata['birthdate']
            gender = metadata['gender']
            nation = metadata['nation']
            author = metadata['author']
            title = metadata['title']
            canonicity = metadata['canonicity']
            pubname = metadata['pubname']
            allwords = self.training.volsizes[volid]
            logistic = self.predictions.get(volid, 'untested')
            realclass = self.training.volumes.label[volid]
            outrow = [volid, reviewed, obscure, pubdate, birthdate,
                      gender, nation, allwords, logistic, author,
                      title, pubname, canonicity, realclass]
            allvolumes.append(outrow)

        return allvolumes

    def _full_coefficients(self):
        test_indices = self.training.test_indices
        test_indices = test_indices if test_indices else 0
        trainingset, yvals, testset = modelingprocess.sliceframe(
            self.data, self.training.classvector,
            self.training.dont_train, test_indices
        )

        newmodel = LogisticRegression(C=self.regularization,
                                      penalty=self.penalty)
        trainingset, means, stdevs = normalizearray(trainingset,
                                                    self.training.usedate)
        if self.verbose:
            print('Reporting coefficients for {} features based on {} '
                  'training samples'.format(len(self.training.vocablist),
                                            len(trainingset)))
            print('Using {} penalty, with {} regularization'.format(
                self.penalty, self.regularization
            ))
        newmodel.fit(trainingset, yvals)
        coefficients = newmodel.coef_[0] * 100
        coefficientuples = zip(coefficients,
                               (coefficients / np.array(stdevs)),
                               self.training.vocablist + ['pub.date'])

        return sorted(coefficientuples)

    def _predict(self):
        raise NotImplementedError

class LeaveOneOutModel(_ValidationModel):
    def _predict(self):
        model_args = list()
        for i, volid in enumerate(self.training.volumes):
            listtoexclude = self.training.authormatches[i]
            arg_set = (self.data, self.training.classvector, listtoexclude, i,
                       self.training.usedate, self.regularization, self.penalty)
            model_args.append(arg_set)

        # Now do leave-one-out predictions.
        print('Beginning multiprocessing.')

        pool = Pool(processes=8)
        res = pool.map_async(modelingprocess.model_one_volume, model_args)

        # After all files are processed, write metadata, errorlog, and counts of
        # phrases.
        res.wait()
        resultlist = res.get()

        assert len(resultlist) == len(self.training.volumes)

        predictions = dict()
        for i, volid in enumerate(self.training.volumes):
            predictions[volid] = resultlist[i]

        pool.close()
        pool.join()

        print('Multiprocessing concluded.')
        print()

        return predictions

class TestModel(_ValidationModel):
    def _predict(self):
        test_indices = self.training.test_indices
        test_indices = test_indices if test_indices else 0
        trainingset, yvals, testset = modelingprocess.sliceframe(
            self.data, self.training.classvector,
            self.training.dont_train, test_indices
        )
        model = LogisticRegression(C=self.regularization,
                                   penalty=self.penalty)
        trainingset, means, stdevs = normalizearray(trainingset,
                                                    self.training.usedate)
        model.fit(trainingset, yvals)

        testset = (testset - means) / stdevs
        results = model.predict_proba(testset)[:, 1]

        predictions = dict()
        for i, pred in zip(self.training.test_indices, results):
            volid = self.training.volumes[i]
            predictions[volid] = pred
        return predictions

class FeatureSelectModel(TestModel):
    def __init__(self, training, penalty, regularization,
                 verbose=False, granularity=4):
        self.granularity = granularity
        super().__init__(training, penalty, regularization, verbose)

    def _predict(self):
        predictions = {}
        iterations = 0
        all_best_words = set()
        while self.training.next_testdata():
            print('Selection batch {}'.format(iterations))
            grid = GridSearch(self.training, 1, -2, self.granularity)
            self.training.set_vocablist(grid.best_words)
            all_best_words.update(grid.best_words)
            testmodel = TestModel(self.training, self.penalty,
                                  self.regularization, verbose=True)
            predictions.update(testmodel.predictions)
            print()
            print('Features selected:')
            print(' '.join(grid.best_words))
            print('Batch Accuracy:')
            print(testmodel.accuracy())
            print('Cumulative Accuracy:')
            print(self.accuracy(predictions))
            print()
            self.training.set_vocablist()
            iterations += 1
        print()
        self.training.set_vocablist(grid.best_words)
        return predictions

def grid_multiprocessing_model(args):
    training, penalty, reg = args
    model = TestModel(training, penalty, reg)
    coefficients = {word: (coef, normed)
                    for coef, normed, word in model.coefficientuples}
    sys.stdout.write('.')
    sys.stdout.flush()
    return coefficients

def poolmap(func, seq):
    pool = Pool(processes=16)
    result = pool.map(func, seq)
    pool.close()  # Otherwise processes build up and trigger a
    pool.join()   # too-many-files-open error.
    return result

class GridSearch(object):
    def __init__(self, training, low_exp, high_exp, steps,
                 use_l2=False, fileoutput=False, verbose=False, ticks=True):

        # Stored as a list to enable multiple penalties in the future:
        self.penalties = ['l2'] if use_l2 else ['l1']
        self.regs = self.exp_steps(low_exp, high_exp, steps)

        self.out_filename = 'gridpredictions_p-{}_r-{}.csv'.format
        self.coef_filename = 'gridpredictions_p-{}_r-{}.coefs.csv'.format
        self.gridword_filename = 'gridwords_{}.csv'.format
        self.training = training
        self.verbose = verbose
        self.ticks = ticks
        self.fileoutput = fileoutput

        if verbose:
            self.grid_information()
        self.best_words = self.grid_search()[0]  # Corresponds to 'l1' features

    def grid_information(self):
        model_descr = ' {:<15} {}'.format
        print("Generating models for the following penalty-parameter pairs:")
        print()
        print("Penalty Type    Regularization Parameter")
        print("\n".join(model_descr(p, r)
                        for p in self.penalties for r in
                        self.regs))
        print()

    def exp_steps(self, start_exp, end_exp, granularity):
        """A convenience function that calculates steps in log10 space. Examples:

            exp_steps(0, 3, 1) -> [1, 10, 100]
            exp_steps(0, 3, 2) -> [1, sqrt(10), 10, sqrt(100), 100, sqrt(1000)]
            exp_steps(0, -4, 1) -> [1, 0.1, 0.01, 0.001]
        """
        direction = -1 if start_exp > end_exp else 1
        roundto = 1 - min(min(start_exp, end_exp), 0)
        exps = range(start_exp * granularity, end_exp * granularity, direction)
        steps = [round(10 ** (e / granularity), roundto) for e in exps]
        steps.append(10 ** (end_exp))
        return steps

    def grid_search(self):
        args = [(self.training, p, r)
                for p in self.penalties
                for r in self.regs]

        # coefs = map(self.search_model, args)
        coefs = poolmap(grid_multiprocessing_model, args)

        all_coefs = {}
        for (_training_unused, pen, reg), coefs in zip(args, coefs):
            all_coefs[pen, reg] = coefs
        return self.save_word_vectors(all_coefs)

    def search_model(self, args):
        training, penalty, reg = args
        model = TestModel(training, penalty, reg)
        if self.verbose:
            print()
            print('Penalty: {}     Regularization: {}'.format(penalty, reg))
            print()
        elif self.ticks:
            sys.stdout.write('.')
            sys.stdout.flush()

        # TODO: Implement file output for slow grid search.
        # if self.verbose or self.fileoutput:
        #     raw = model.accuracy()
        #     tilt = diachronic_tilt(model.allvolumes, 'linear', [])
        # if self.fileoutput:
        #     write_coefficients(model.coefficientuples,
        #                        self.out_filename(penalty, reg))
        #     Write accuracy and tilt accuracy out to a file too.
        # if self.verbose:
        #     display_tilt_accuracy(raw, tilt)

        coefficients = {word: (coef, normed)
                        for coef, normed, word in model.coefficientuples}
        return coefficients

    def save_word_vectors(self, all_coefs):
        # create word vectors over dimensions of regularization space
        words = self.training.vocablist
        vectors = self.grid_vectors(words, all_coefs)

        reduced_features = []
        for p in self.penalties:
            output_data = self.sort_by_covar(vectors[p])
            output_words = output_data.dtype.names
            header = ', '.join(output_words)
            np.savetxt(self.gridword_filename(p), output_data,
                       header=header, fmt='%8.5f')
            best = [w for w in output_words if output_data[w][-1] < 0.0005]

            reduced_features.append(best)
        return reduced_features

    def grid_vectors(self, words, all_coefs):
        """Create a numpy record array, where `a.l1.fear` refers to the L1 penalty
        coefficients for the word 'fear' across all regularization parameter
        values, represented as percentages of the total L1 penalty.
        """
        word_dtype = [('model_regularization_param', float)]
        word_dtype.extend([(w, float) for w in words])
        vectors = np.recarray(len(self.regs), dtype=[('l1', word_dtype),
                                                     ('l2', word_dtype)])

        for pen in self.penalties:
            for reg_ix, reg in enumerate(self.regs):
                vectors[pen]['model_regularization_param'][reg_ix] = reg
                word_coefs = all_coefs[pen, reg]
                for word in words:
                    coef, _normed_unused = word_coefs.get(word, (0, 0))
                    coef = float(coef)
                    coef = 0 if math.isnan(coef) or math.isinf(coef) else coef
                    vectors[pen][word][reg_ix] = coef

        # Calculate the effective L1 penality for each regularization value
        # and normalize, such that each weight is represented as a percentage
        # of total model penalty. The L2 penalty is convex, so this approach
        # can be used for it too. It will distort proportions,
        # but will not affect the order of the results.
        for pen in self.penalties:
            pen_sum = np.sum(np.abs(vectors[pen][word]) for word in words)
            pen_sum[pen_sum == 0] = 1  # Avoid div by zero.
            for word in words:
                vectors[pen][word] /= pen_sum
        return vectors

    def covar(self, a, b):
        return np.dot(a - a.mean(), b - b.mean()) / (len(a) - 1)

    def sort_by_covar(self, vecs):
        reg = vecs['model_regularization_param']
        cov_words = [(self.covar(reg, np.abs(vecs[w])), w)
                     for w in vecs.dtype.names]
        cov_words = [(c, w) for c, w in cov_words if c != 0]
        cov_words.sort()

        sorted_words = [w for c, w in cov_words]
        out_vecs = np.recarray(len(vecs) + 1,
                               dtype=[(w, float) for w in sorted_words])

        for cov, w in cov_words:
            out_v = list(vecs[w])
            out_v.append(float(cov))
            out_vecs[w] = out_v
        return out_vecs

def grid_feature_select(settings, test_step=1, grid_granularity=4):
    sourcefolder, extension, classpath, outputpath = settings.paths
    category2sorton, positive_class, datetype, numfeatures, \
        reg, penalty = settings.classifyconditions
    pastthreshold, futurethreshold = settings.thresholds

    volumes = VolumeMeta(sourcefolder, extension, classpath,
                         settings.exclusions, category2sorton, positive_class)
    training = TrainingData(volumes, pastthreshold, futurethreshold,
                            datetype, numfeatures, test_step=test_step)
    model = FeatureSelectModel(training, penalty, reg,
                               verbose=False, granularity=grid_granularity)
    write_output_rows(model.header, model.allvolumes, outputpath)
    write_coefficients(model.coefficientuples, outputpath)

    return model.accuracy(), model.allvolumes, model.coefficientuples

def create_model(paths, exclusions, thresholds, classifyconditions):
    ''' This is the main function in the module.
    It can be called externally; it's also called
    if the module is run directly.
    '''
    verbose = False
    sourcefolder, extension, classpath, outputpath = paths
    category2sorton, positive_class, datetype, \
        numfeatures, regularization, penalty = classifyconditions
    pastthreshold, futurethreshold = thresholds

    volumes = VolumeMeta(sourcefolder, extension, classpath, exclusions,
                         category2sorton, positive_class)
    training = TrainingData(volumes, pastthreshold, futurethreshold,
                            datetype, numfeatures)

    model = LeaveOneOutModel(training, penalty, regularization)
    write_output_rows(model.header, model.allvolumes, outputpath)
    write_coefficients(model.coefficientuples, outputpath)
    if verbose:
        model.display_coefficients()

    return model.accuracy(), model.allvolumes, model.coefficientuples

def diachronic_tilt(allvolumes, modeltype, datelimits, plot=False):
    ''' Takes a set of predictions produced by a model that knows nothing about date,
    and divides it along a line with a diachronic tilt. We need to do this in a way
    that doesn't violate crossvalidation. I.e., we shouldn't "know" anything
    that the model didn't know. We tried a couple of different ways to do this, but
    the simplest and actually most reliable is to divide the whole dataset along a
    linear central trend line for the data!
    '''

    date = [vol[3] for vol in allvolumes]
    logistic = [vol[8] for vol in allvolumes]
    classvector = [vol[13] for vol in allvolumes]

    d_l_c = list(zip(date, logistic, classvector))
    reviewedx = [d for d, l, c in d_l_c if c == 1 and l != 'untested']
    reviewedy = [l for d, l, c in d_l_c if c == 1 and l != 'untested']
    randomx = [d for d, l, c in d_l_c if c != 1 and l != 'untested']
    randomy = [l for d, l, c in d_l_c if c != 1 and l != 'untested']
    untestedreviewedx = [d for d, l, c in d_l_c if c == 1 and l == 'untested']
    untestedrandomx = [d for d, l, c in d_l_c if c != 1 and l == 'untested']

    xmin = min(d for d, l, c in d_l_c)
    xmax = max(d for d, l, c in d_l_c)
    ymin = min(l for d, l, c in d_l_c if l != 'untested')
    ymax = max(l for d, l, c in d_l_c if l != 'untested')
    yrange = max(ymax - ymin, 0.01)
    untestedreviewedy = [ymin - yrange * 0.03] * len(untestedreviewedx)
    untestedrandomy = [ymin - yrange * 0.06] * len(untestedrandomx)

    if plot:
        plt.clf()
        plt.axis([xmin - 2, xmax + 2, ymin - yrange * 0.09, ymax + yrange * 0.02])
        plt.plot(reviewedx, reviewedy, 'ro')
        plt.plot(randomx, randomy, 'k+')
        plt.plot(untestedreviewedx, untestedreviewedy, 'bo')
        plt.plot(untestedrandomx, untestedrandomy, 'b+')

    test_xyc = zip(*[(d, l, c) for d, l, c in zip(date, logistic, classvector)
                     if l != 'untested'])
    x, y, trueclass = map(np.array, test_xyc)

    if modeltype == 'logistic':
        # all this is DEPRECATED
        print("Hey, you're attempting to use the logistic-tilt option")
        print("that we deactivated. Go in and uncomment the code.")

        # DEPRECATED
        # if len(datelimits) == 2:
        #     # In this case we construct a subset of data to model on.
        #     pastthreshold, futurethreshold = datelimits
        #     d_l_c = [(d, l, c)
        #              for d, l, c in zip(date, logistic, classvector)
        #              if pastthreshold <= d <= futurethreshold and
        #              l != 'untested']
        # else:
        #     d_l_c = [(d, l, c)
        #              for d, l, c in zip(date, logistic, classvector)
        #              if l != 'untested']

        # data = pd.DataFrame([(l, d) for d, l, c in d_l_c])
        # responsevariable = [c for d, l, c in d_l_c]

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

    if plot:
        plt.plot(x, p(x), "b-")
        plt.show(block=False)
        plt.pause(0.1)

    dividingline = intercept + (x * slope)
    predicted_as_reviewed = (y > dividingline)
    really_reviewed = (trueclass == 1)

    correct = sum(predicted_as_reviewed == really_reviewed)
    return correct / len(trueclass)

def display_tilt_accuracy(rawaccuracy, tiltaccuracy):
    msg = 'If we divide the dataset with a horizontal line at 0.5, accuracy is:'
    print(msg, str(rawaccuracy))
    msg = "Divided with a line fit to the data trend, it's"
    print(msg, str(tiltaccuracy))

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
    tiltaccuracy = diachronic_tilt(allvolumes, 'linear', [])
    display_tilt_accuracy(rawaccuracy, tiltaccuracy)
