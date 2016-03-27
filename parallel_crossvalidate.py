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
from util import metafilter

import matplotlib
# matplotlib.use("TkAgg")  # Use geenric (portable) plotting backend.
matplotlib.use("Agg")  # Get rid of the annoying Python rocket but lose plots.
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

def normalizearray(featurearray):
    '''Normalizes an array by centering on means and
    scaling by standard deviations. Also returns the
    means and standard deviations for features, so that
    they can be pickled.
    '''

    numinstances, numfeatures = featurearray.shape
    means = list()
    stdevs = list()
    for featureidx in range(numfeatures):

        thiscolumn = featurearray.iloc[:, featureidx]
        thismean = np.mean(thiscolumn)

        thisstdev = np.std(thiscolumn)

        means.append(thismean)
        stdevs.append(thisstdev)
        featurearray.iloc[:, featureidx] = \
            (thiscolumn - thismean) / thisstdev

    return featurearray, means, stdevs

def sliceframe(dataframe, yvals, excludedrows, testrow):
    exrowset = set(excludedrows)
    newyvals = [y for i, y in enumerate(yvals) if i not in exrowset]

    trainingset = dataframe.drop(dataframe.index[excludedrows])

    newyvals = np.array(newyvals)
    testset = dataframe.iloc[testrow]

    return trainingset, newyvals, testset

def model_one_volume(model_args):
    data, classvector, listtoexclude, i, regularization, penalty = model_args

    trainingset, yvals, testset = sliceframe(data, classvector,
                                             listtoexclude, i)
    newmodel = LogisticRegression(C=regularization, penalty=penalty)
    trainingset, means, stdevs = normalizearray(trainingset)
    newmodel.fit(trainingset, yvals)

    testset = (testset - means) / stdevs
    prediction = newmodel.predict_proba(testset.reshape(1, -1))[0][1]
    if i % 50 == 0:
        print(i)
    # print(str(i) + "  -  " + str(len(listtoexclude)))
    return prediction

def not_pred(pred, value, default=None):
    return value if not pred(value) else default

def not_none(value, default=None):
    return not_pred(lambda v: v is None, value, default)

class VolumeMeta(object):
    """A class representing a collection of HathiTrust volumes with
    accompanying metadata. All attributes should be treated as
    read-only.
    """
    def __init__(self, sourcefolder, extension, classpath, exclusions,
                 category2sorton, positive_class):

        files = [f for f in os.listdir(sourcefolder) if f.endswith(extension)]
        all_ids = [f.rpartition(extension)[0] for f in files]

        # We gather data for all volumes in the directory, but...
        exif, exifnot, exbelow, exabove, sizecap = exclusions
        self.meta = metafilter.get_metadata(
            classpath, all_ids, exif, exifnot, exbelow, exabove
        )
        self.label = metafilter.classlabels(
            self.meta, category2sorton, positive_class, sizecap
        )
        self.path = {vid: os.path.join(sourcefolder, filename)
                     for vid, filename in zip(all_ids, files)}
        self.wordcount = {vid: dict(self._volume_wordcount(vid))
                          for vid in all_ids}
        self.totalcount = {vid: sum(self.wordcount[vid].values())
                           for vid in all_ids}

        # ...this determines which volumes are "in" the collection.
        self.ordered_ids = [vid for vid in all_ids if vid in self.label]

        self.metavector = [self.meta[vid] for vid in self]
        self.labelvector = [self.label[vid] for vid in self]
        self.pathvector = [self.path[vid] for vid in self]
        self.wordcountvector = [self.wordcount[vid] for vid in self]
        self.totalcountvector = [self.totalcount[vid] for vid in self]
        self._groupcache = {}

    def __iter__(self):
        return iter(self.ordered_ids)

    def __len__(self):
        return len(self.ordered_ids)

    def __getitem__(self, index):
        return self.ordered_ids[index]

    def itermeta(self, key):
        """An alias for iterating over a given metadata value in the
        order defiend by ``self.ordered_ids``. The key names defined
        here are::
            'meta': The complete dictionary of metadata for each volume.
            'path': The path to the wordcount file for each volume.
            'label': The class label for each volume.
            'wordcount': The dictionary of wordcounts for each volume.
            'totalcount': The total number of word tokens in each volume.
        All word counts here are token counts -- tokens of particular
        word types in the case of ``wordcount``, and total number of
        tokens in the case of ``totalcount``.

        Other key names are defined by metadata headers via the
        ``get_metadata`` function called above.
        """
        key = 'meta' if key is None else key
        if key in ('meta', 'path', 'label', 'wordcount', 'totalcount'):
            return iter(getattr(self, key + 'vector'))
        else:
            return (self.meta[oid][key] for oid in self)

    def zipmeta(self, *keys):
        """An alias for iterating over tuples of metadata values
        in the order defined by ``self.ordered_ids``. The volume
        ID is always included, and if no key is passed it is
        zipped with the entire ``meta`` dictionary for each volume.
        """
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

    def _volume_wordcount(self, volid):
        """Return a sequence of (word, count) pairs for a given volume."""
        volpath = self.path[volid]
        with open(volpath, encoding='utf-8') as f:
            rows = (l.strip().split('\t') for l in f)
            pairs = (row for row in rows if len(row) == 2)
            return [(word, int(count)) for word, count in pairs]

    def infer_date(self, volid, datetype):
        return infer_date(self.meta[volid], datetype)

class TrainingData(object):
    """A class that abstracts away the details of managing feature
    sets and training data. All attributes should be treated as
    read-only. Public methods are provided for holding out "test"
    data (e.g. for k-fold cross validation) and for changing feature
    sets (i.e. via vocablist).

    Public methods: ``next_testdata``, ``set_vocablist``.
    """
    def __init__(self, volumes, pastthreshold, futurethreshold,
                 datetype, numfeatures, kfold_step=None, vocab=None):
        self.volumes = volumes
        self.pastthreshold = pastthreshold
        self.futurethreshold = futurethreshold
        self.datetype = datetype

        self.test_start = self.test_end = 0
        self.kfold_step = kfold_step
        self.shuffled_indices = self._shuffled_data()
        self.dont_train = self._dont_train()
        self.authormatches = self._authormatches()

        self.word_doc_counts = self._word_doc_counts()
        self.numfeatures = numfeatures

        # These will be set in ``set_vocablist``:
        self.vocablist = self.voldata = self.dataframe = None
        self.set_vocablist(vocab)

    def _shuffled_data(self):
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

        # Just return indices for now; IDs weren't being used.
        return shuf_indices

    def _dont_train(self, drop_random=0):
        drop_random = (1 if drop_random > 1 else
                       0 if drop_random < 0 else
                       drop_random)
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
            elif random.random() < drop_random:
                dont_train.append(i)

        return dont_train

    def _authormatches(self):
        # Since we are going to use these indexes to exclude rows,
        # we also add all the ids in dont_train to every volume.
        combined = (self.dont_train + ai
                    for ai in self.volumes.groupmeta('author'))
        return [sorted(set(c)) for c in combined]

    def next_testdata(self):
        if self.kfold_step is None:
            return False

        self.test_end += self.kfold_step
        # Because test_start and test_end both start at 0...
        self.test_start = self.test_end - self.kfold_step
        self.dont_train = self._dont_train()
        self.authormatches = self._authormatches()

        # Tell caller whether any test data remains.
        return bool(self.test_indices)

    def drop_training_subset(self, fraction):
        """Here, we artificallly reduce the size of the training set by
        taking a random subset. This can be useful for plotting learning
        curves and for experimenting with many different combinations
        of data.
        """
        self.dont_train = self._dont_train(fraction)
        self.authormatches = self._authormatches()

    @property
    def test_indices(self):
        return self.shuffled_indices[self.test_start:self.test_end]

    def _word_doc_counts(self):
        """Count the number of times each word type appears in a
        document in the corpus. Ignore token counts."""
        word_doc_counts = Counter()
        for volid, wc in self.volumes.zipmeta('wordcount'):
            date = self.volumes.infer_date(volid, self.datetype)
            if self.pastthreshold <= date <= self.futurethreshold:
                # Ignore the counts for now; for feature selection, we
                # just want the number of documents that contain the word.
                features = (word for word in wc
                            if len(word) > 0 and word[0].isalpha())
                word_doc_counts.update(features)
        return word_doc_counts

    def set_vocablist(self, vocab=None):
        if vocab is None:
            vocab = [x[0] for x in
                     self.word_doc_counts.most_common(self.numfeatures)]

        self.vocablist = vocab
        self.voldata = self._voldata()
        self.dataframe = pd.DataFrame(self.voldata)

    def _feature_vector(self, wordcounts):
        features = [wordcounts.get(w, 0) for w in self.vocablist]
        return np.array(features, dtype=np.float64)

    def _voldata(self):
        # Goldsotne expressed concern about the (totalcount + 0.001) below,
        # but I'm leaving it in until I understand why it's there. -SE
        voldata = []
        for volid, wc, tc in self.volumes.zipmeta('wordcount', 'totalcount'):
            features = self._feature_vector(self.volumes.wordcount[volid])
            features /= (self.volumes.totalcount[volid] + 0.001)
            voldata.append(features)
        return voldata

    @property
    def classvector(self):
        return self.volumes.labelvector

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
    def __init__(self, training, penalty, regularization, verbose=False,
                 feature_selector=None, poolsize=8):
        self.training = training
        self.regularization = regularization
        self.penalty = penalty
        self.verbose = verbose
        self.feature_selector = feature_selector or (lambda x: None)
        self.poolsize = poolsize
        self.predict()

    def predict(self, regularization=None, penalty=None):
        self.regularization = not_none(regularization, self.regularization)
        self.penalty = not_none(penalty, self.penalty)
        self.predictions = self._predict()
        self.allvolumes = self._make_output_rows()
        self.coefficientuples = self._full_coefficients()

    @property
    def data(self):
        return self.training.dataframe

    def accuracy(self, predictions=None):
        predictions = predictions or self.predictions
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

    def write_coefficients(self, outputpath):
        return write_coefficients(self.coefficientuples, outputpath)

    def write_output_rows(self, outputpath):
        return write_output_rows(self.header, self.allvolumes, outputpath)

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
            allwords = self.training.volumes.totalcount[volid]
            logistic = self.predictions.get(volid, 'untested')
            realclass = self.training.volumes.label[volid]
            outrow = [volid, reviewed, obscure, pubdate, birthdate,
                      gender, nation, allwords, logistic, author,
                      title, pubname, canonicity, realclass]
            allvolumes.append(outrow)

        return allvolumes

    def _full_coefficients(self):
        test_indices = self.training.test_indices or 0
        trainingset, yvals, testset = sliceframe(
            self.data, self.training.classvector,
            self.training.dont_train, test_indices
        )

        newmodel = LogisticRegression(C=self.regularization,
                                      penalty=self.penalty)
        trainingset, means, stdevs = normalizearray(trainingset)
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
                       self.regularization, self.penalty)
            model_args.append(arg_set)

        # Now do leave-one-out predictions.
        pool = Pool(processes=self.poolsize)
        res = pool.map_async(model_one_volume, model_args)

        # After all files are processed, write metadata, errorlog,
        # and counts of phrases.
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
        test_indices = self.training.test_indices or 0
        trainingset, yvals, testset = sliceframe(
            self.data, self.training.classvector,
            self.training.dont_train, test_indices
        )
        model = LogisticRegression(C=self.regularization,
                                   penalty=self.penalty)
        trainingset, means, stdevs = normalizearray(trainingset)
        model.fit(trainingset, yvals)

        testset = (testset - means) / stdevs
        results = model.predict_proba(testset)[:, 1]

        predictions = dict()
        for i, pred in zip(self.training.test_indices, results):
            volid = self.training.volumes[i]
            predictions[volid] = pred
        return predictions

# Would like to try training on combinations of features.

# class TestModelNew(_ValidationModel):
#     def _predict(self):
#         test_indices = self.training.test_indices or 0
#         trainingset, yvals, testset = sliceframe(
#             self.data, self.training.classvector,
#             self.training.dont_train, test_indices
#         )
#         model = LogisticRegression(C=self.regularization,
#                                    penalty=self.penalty)
#         trainingset, means, stdevs = normalizearray(trainingset)
#         trainingset = np.as_array(trainingset)
#         print(trainingset)
#         model.fit(trainingset, yvals)
#
#         testset = (testset - means) / stdevs
#         results = model.predict_proba(testset)[:, 1]
#
#         predictions = dict()
#         for i, pred in zip(self.training.test_indices, results):
#             volid = self.training.volumes[i]
#             predictions[volid] = pred
#         return predictions

class FeatureSelectModel(_ValidationModel):
    def _predict(self):
        predictions = {}
        iterations = 0
        all_features = Counter()
        while self.training.next_testdata():
            print('Feature selection validation batch {}'.format(iterations))

            selected_features = self.feature_selector(self.training)
            self.training.set_vocablist(selected_features)
            features = self.training.vocablist  # TrainingData is slightly
            all_features.update(features)       # opinionated about features.

            testmodel = TestModel(self.training, self.penalty,
                                  self.regularization, verbose=True)
            predictions.update(testmodel.predictions)
            print()
            print('Features selected:')
            print(' '.join(features))
            print('Batch Accuracy:')
            print(testmodel.accuracy())
            print('Cumulative Accuracy:')
            print(self.accuracy(predictions))
            print()
            self.training.set_vocablist()
            iterations += 1
        print()
        self.training.set_vocablist([w for w in all_features
                                     if all_features[w] > 1])
        return predictions

class SimpleL1Search(object):
    def __init__(self, training=None, regularization=0.01, nfeatures=None):
        self.training = training
        self.regularization = regularization
        self.nfeatures = nfeatures

    def __call__(self, training=None):
        self.training = training or self.training
        if self.training is None:
            return None

        model = TestModel(self.training, 'l1', self.regularization)
        topc = sorted((coef, word) for coef, normed, word
                      in model.coefficientuples)
        features = [w for c, w in topc if c != 0]

        if self.nfeatures is None or self.nfeatures >= len(features):
            return features
        else:
            bottom = self.nfeatures / 2
            top = self.nfeatures - bottom
            return features[:bottom] + features[top:]

class GridSearch(object):
    def __init__(self, training=None,
                 start_exp=1, end_exp=-2, granularity=4, n_features=200,
                 dropout_trials=0, dropout_fraction=None,
                 use_l2=False,
                 fileoutput=False, verbose=False, ticks=True, poolmax=8):
        self.penalty = 'l2' if use_l2 else 'l1'
        self.regs = self.exp_steps(
            start_exp or 1,
            end_exp or -2,
            granularity or 4
        )

        if dropout_trials > 0:
            dropout_fraction = not_none(dropout_fraction, 0.10)
            self.dropout_fraction = min(max(dropout_fraction, 0), 1)
        else:
            self.dropout_fraction = 0

        self.trials = dropout_trials if dropout_trials > 0 else 1
        self.n_features = n_features

        self.poolsize = min(granularity * 4, poolmax)
        self.out_filename = 'gridpredictions_p-{}_r-{}.csv'.format
        self.coef_filename = 'gridpredictions_p-{}_r-{}.coefs.csv'.format
        self.gridword_filename = 'gridwords_{}_{}.csv'.format
        self.verbose = verbose
        self.ticks = ticks
        self.fileoutput = fileoutput

        self.training = training
        self.best_words = None

    # It's a little strange to have a callable class automatically save a
    # file as a side-effect, but it's easy to change this default now.
    def __call__(self, training=None, save_file=False):
        self.training = training or self.training
        if self.training is None:
            return None

        feature_covars = {}
        for i in range(self.trials):
            sys.stdout.write('Grid search trial {}: '.format(i))
            sys.stdout.flush()
            if self.dropout_fraction:
                self.training.drop_training_subset(self.dropout_fraction)
            vectors = self.grid_search()
            output_data = self.sort_by_covar(vectors[self.penalty])
            for feature in output_data.dtype.names:
                covar = feature_covars.get(feature, 0)
                covar += output_data[feature][-1] / self.trials
                feature_covars[feature] = covar
            print()

        if save_file:
            self.save_grid_vectors(vectors)

        return sorted(feature_covars, key=feature_covars.get)[:self.n_features]

    def grid_information(self):
        model_descr = ' {:<15} {}'.format
        print("Generating models for the following penalty-parameter pairs:")
        print()
        print("Penalty Type    Regularization Parameter")
        print("\n".join(model_descr(self.penalty, r)
                        for r in self.regs))
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
        if self.training is None:
            return [None]  # Because grid search is expected to return a list.

        if self.verbose:
            self.grid_information()

        args = [(self.training, self.penalty, r)
                for r in self.regs]

        coefs = self.poolmap(self.search_model, args, self.poolsize)

        all_coefs = {}
        for (_training_unused, pen, reg), coefs in zip(args, coefs):
            all_coefs[pen, reg] = coefs

        return self.grid_vectors(all_coefs)

    @staticmethod
    def poolmap(func, seq, poolsize):
        pool = Pool(processes=poolsize)
        result = pool.map(func, seq, chunksize=1)
        pool.close()  # Otherwise processes build up and trigger a
        pool.join()   # too-many-files-open error.
        return result

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

        if self.verbose or self.fileoutput:
            raw = model.accuracy()
            tilt = diachronic_tilt(model.allvolumes, [])
        if self.fileoutput:
            model.write_coefficients(self.out_filename(penalty, reg))
            # TODO: Write accuracy and tilt accuracy out to a file too.
        if self.verbose:
            display_tilt_accuracy(raw, tilt)

        coefficients = {word: (coef, normed)
                        for coef, normed, word in model.coefficientuples}
        return coefficients

    def save_grid_vectors(self, vectors, trial):
        output_data = self.sort_by_covar(vectors[self.penalty])
        output_words = output_data.dtype.names
        header = ', '.join(output_words)
        np.savetxt(self.gridword_filename(self.penalty, trial), output_data,
                   header=header, fmt='%8.5f')

    def grid_vectors(self, all_coefs):
        """Create a numpy record array, where `a.l1.fear` refers to the L1 penalty
        coefficients for the word 'fear' across all regularization parameter
        values, represented as percentages of the total L1 penalty.
        """
        words = self.training.vocablist
        word_dtype = [('model_regularization_param', float)]
        word_dtype.extend([(w, float) for w in words])
        vectors = np.recarray(
            len(self.regs),
            dtype=[(self.penalty, word_dtype)]
        )

        for reg_ix, reg in enumerate(self.regs):
            vectors[self.penalty]['model_regularization_param'][reg_ix] = reg
            word_coefs = all_coefs[self.penalty, reg]
            for word in words:
                coef, _normed_unused = word_coefs.get(word, (0, 0))
                coef = float(coef)
                coef = 0 if math.isnan(coef) or math.isinf(coef) else coef
                vectors[self.penalty][word][reg_ix] = coef

        # Calculate the effective L1 penality for each regularization value
        # and normalize, such that each weight is represented as a percentage
        # of total model penalty. The L2 penalty is convex, so this approach
        # can be used for it too. It will distort proportions,
        # but will not affect the order of the results.
        pen_sum = np.sum(np.abs(vectors[self.penalty][word]) for word in words)
        pen_sum[pen_sum == 0] = 1  # Avoid div by zero.
        for word in words:
            vectors[self.penalty][word] /= pen_sum
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

def diachronic_tilt(allvolumes, datelimits, plot=False):
    """Takes a set of predictions produced by a model that knows
    nothing about date, and divides it along a line with a diachronic
    tilt. We need to do this in a way that doesn't violate crossvalidation.
    I.e., we shouldn't "know" anything that the model didn't know. We
    tried a couple of different ways to do this, but the simplest and
    actually most reliable is to divide the whole dataset along a linear
    central trend line for the data!
    """

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
        plt.axis([xmin - 2, xmax + 2,
                  ymin - yrange * 0.09, ymax + yrange * 0.02])
        plt.plot(reviewedx, reviewedy, 'ro')
        plt.plot(randomx, randomy, 'k+')
        plt.plot(untestedreviewedx, untestedreviewedy, 'bo')
        plt.plot(untestedrandomx, untestedrandomy, 'b+')

    test_xyc = zip(*[(d, l, c) for d, l, c in zip(date, logistic, classvector)
                     if l != 'untested'])
    x, y, trueclass = map(np.array, test_xyc)

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
    # If this class is called directly, it creates a single model using the
    # default settings from `replicate.py`. Ordinarily, we wouldn't create a
    # circular import this way, but because this only happens when the script
    # is run directly, it's OK.

    from replicate import Settings, leave_one_out_model
    leave_one_out_model(Settings())
