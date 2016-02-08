# modelingprocess.py

import numpy as np
from models import LogisticRegression

def sliceframe(dataframe, yvals, excludedrows, testrow):
    exrowset = set(excludedrows)
    newyvals = [y for i, y in enumerate(yvals) if i not in exrowset]

    trainingset = dataframe.drop(dataframe.index[excludedrows])

    newyvals = np.array(newyvals)
    testset = dataframe.iloc[testrow]

    return trainingset, newyvals, testset

def normalizearray(featurearray):
    '''Normalizes an array by centering on means and
    scaling by standard deviations. Also returns the
    means and standard deviations for features.
    '''

    numinstances, numfeatures = featurearray.shape
    means = list()
    stdevs = list()
    for featureidx in range(numfeatures):

        thiscolumn = featurearray.iloc[ : , featureidx]
        thismean = np.mean(thiscolumn)

        thisstdev = np.std(thiscolumn)

        means.append(thismean)
        stdevs.append(thisstdev)
        featurearray.iloc[ : , featureidx] = (thiscolumn - thismean) / thisstdev
    return featurearray, means, stdevs

def model_one_volume(model_args):
    data, classvector, listtoexclude, i, regularization, penalty = model_args

    trainingset, yvals, testset = sliceframe(data, classvector, listtoexclude, i)
    newmodel = LogisticRegression(C=regularization, penalty=penalty)
    trainingset, means, stdevs = normalizearray(trainingset)
    newmodel.fit(trainingset, yvals)

    testset = (testset - means) / stdevs
    prediction = newmodel.predict_proba(testset.reshape(1, -1))[0][1]
    if i % 50 == 0:
        print(i)
    # print(str(i) + "  -  " + str(len(listtoexclude)))
    return prediction
