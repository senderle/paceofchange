# My goal in this script is to package up the modeling and
# evaluation processes we actually ran to produce the article,
# so they can be replicated without a lot of fuss.

import parallel_crossvalidate as pc
import sys
from pprint import pprint
from numpy import (recarray as np_recarray,
                   savetxt as np_savetxt,
                   abs as np_abs,
                   sum as np_sum)

class Settings(object):
    # REGULARIZATION
    penalty = 'l2'
    regularization = 0.00007

    # PATHS
    sourcefolder = 'poems/'
    extension = '.poe.tsv'
    classpath = 'poemeta.csv'
    outputpath = 'mainmodelpredictions.csv'

    # EXCLUSIONS

    # We're not using reviews from Tait's.
    # We don't ordinarily include canonical volumes that were not in either sample.
    excludeif = {'pubname': 'TEM',
                 'recept': 'addcannon'}

    excludeifnot = {}
    excludebelow = {'firstpub': 1700}
    excludeabove = {'firstpub': 1950}
    sizecap = 360

    # THRESHOLDS
    # For more historically-interesting kinds of questions, we can limit the part
    # of the dataset that gets TRAINED on, while permitting the whole dataset to
    # be PREDICTED. (Note that we always exclude authors from their own training
    # set; this is in addition to that.) The variables futurethreshold and
    # pastthreshold set the chronological limits of the training set, inclusive
    # of the threshold itself.
    futurethreshold = 1925
    pastthreshold = 1800

    # CLASSIFY CONDITIONS
    positive_class = 'rev'
    category2sorton = 'reviewed'
    datetype = 'firstpub'
    numfeatures = 3200

    @property
    def paths(self):
        return (self.sourcefolder, self.extension,
                self.classpath, self.outputpath)

    @property
    def exclusions(self):
        return (self.excludeif, self.excludeifnot, self.excludebelow,
                self.excludeabove, self.sizecap)

    @property
    def thresholds(self):
        return self.pastthreshold, self.futurethreshold

    @property
    def classifyconditions(self):
        return (self.category2sorton, self.positive_class, self.datetype,
                self.numfeatures, self.regularization, self.penalty)

def display_tilt_accuracy(rawaccuracy, tiltaccuracy):
    msg = 'If we divide the dataset with a horizontal line at 0.5, accuracy is:'
    print(msg, str(rawaccuracy))
    msg = "Divided with a line fit to the data trend, it's"
    print(msg, str(tiltaccuracy))

def generic_model(s):
    rawaccuracy, allvolumes, coefficientuples = pc.create_model(
        s.paths, s.exclusions, s.thresholds, s.classifyconditions
    )

    tiltaccuracy = pc.diachronic_tilt(allvolumes, 'linear', [])
    display_tilt_accuracy(rawaccuracy, tiltaccuracy)
    return rawaccuracy, tiltaccuracy

def nations(s):
    s.outputpath = 'nationalpredictions.csv'
    s.pastthreshold = 1700
    s.positive_class = 'uk'
    s.category2sorton = 'nation'
    generic_model(s)

def gender(s):
    s.outputpath = 'genderpredictions.csv'
    s.pastthreshold = 1700
    s.positive_class = 'f'
    s.category2sorton = 'gender'
    generic_model(s)

def canon(s):
    s.outputpath = 'canonpredictions.csv'
    del s.excludeif['recept']
    s.sizecap = 450
    generic_model

def halves(s):
    # DO THE FIRST HALF.
    s.outputpath = 'firsthalfpredictions.csv'
    s.excludebelow['firstpub'] = 1800
    s.excludeabove['firstpub'] = 1875
    s.sizecap = 300
    generic_model(s)

    # NOW DO THE SECOND HALF.
    s.outputpath = 'secondhalfpredictions.csv'
    s.excludebelow['firstpub'] = 1876
    s.excludeabove['firstpub'] = 1925
    generic_model(s)

def quarters(s):
    quarteroptions = [('1820-44predictions.csv', 1800, 1844),
                      ('1845-69predictions.csv', 1845, 1869),
                      ('1870-94predictions.csv', 1870, 1894),
                      ('1895-19predictions.csv', 1895, 1925)]
    # I removed the `quarterresults` line here; it seemed to do nothing. --SE

    for outputpath, pastthreshold, futurethreshold in quarteroptions:
        print(pastthreshold)
        s.outputpath = outputpath
        s.pastthreshold = pastthreshold
        s.futurethreshold = futurethreshold

        generic_model(s)
        # I removed the `theseresults` lines here; see above. --SE

def grid(s):
    regs = grid_steps(1, -2, 8)
    penalties = ['l1', 'l2']

    grid_information(penalties, regs)

    out_filename = 'gridpredictions_p-{}_r-{}.csv'.format
    results = []
    for p in penalties:
        for r in regs:
            s.penalty = p
            s.regularization = r
            s.outputpath = out_filename(p, r)
            print()
            print('Penalty: {}     Regularization: {}'.format(p, r))
            print()
            raw, tilt = generic_model(s)
            results.append({'penalty': p,
                            'regularization': r,
                            'rawaccuracy': raw,
                            'tiltaccuracy': tilt})
    print()
    print("Accuracy for all values:")
    pprint(results)
    print()

    coef_filename = 'gridpredictions_p-{}_r-{}.coefs.csv'.format
    words = grid_words(coef_filename('l2', regs[0]))
    vectors = grid_vectors(words, penalties, regs, coef_filename)
    for p in penalties:
        out_gridwords = 'gridwords_{}.csv'.format
        header = ', '.join(vectors[p].dtype.names)
        np_savetxt(out_gridwords(p), vectors[p],
                   header=header, fmt='%8.5f')

def grid_information(penalties, regs):
    model_descr = ' {:<15} {}'.format
    print("Generating models for the following penalty-parameter pairs:")
    print()
    print("Penalty Type    Regularization Parameter")
    print("\n".join(model_descr(p, r) for p in penalties for r in regs))
    print()
    nmodels = len(penalties) * len(regs)
    print("Total: {}".format(nmodels))
    nhours = (nmodels * 5) // 60
    nminutes = (nmodels * 5) % 60
    print("Approximate time requred: {} hours, {} minutes.".format(nhours,
                                                                   nminutes))

def grid_steps(start_exp, end_exp, granularity):
    """A convenience function that calculates steps in log10 space. Examples:

        grid_steps(0, 3, 1) -> [1, 10, 100]
        grid_steps(0, 3, 2) -> [1, sqrt(10), 10, sqrt(100), 100, sqrt(1000)]
        grid_steps(0, -4, 1) -> [1, 0.1, 0.01, 0.001]
    """
    direction = -1 if start_exp > end_exp else 1
    roundto = 1 - min(min(start_exp, end_exp), 0)
    exps = range(start_exp * granularity, end_exp * granularity, direction)
    return [round(10 ** (e / granularity), roundto) for e in exps]

def grid_words(filename, nwords=0):
    """Select the top nwords / 2 and bottom nwords / 2 words based on
    coefficients in a given coefficient csv file with this format:

        word1,normed_coefficient1,coefficient1
        word2,normed_coefficient2,coefficient2
        ...
    """
    with open(filename) as f:
        rows = (line.split(',') for line in f)
        rows = (r for r in rows if len(r) == 3)
        rows = [(float(coef), word) for word, coef, _ in rows]

    nwords = int(nwords) if nwords > 0 else len(rows)
    words = [word for coef, word in sorted(rows)]
    return words[:nwords // 2] + words[-nwords // 2:]

def grid_vectors(words, penalties, regs, out_filename):
    """Create a numpy record array, where `a.l1.fear` refers to the L1 penalty
    coefficients for the word 'fear' across all regularization parameter
    values, represented as percentages of the total L1 penalty."""
    word_dtype = [('model_regularization_param', float)]
    word_dtype.extend([(w, float) for w in words])
    vectors = np_recarray(len(regs), dtype=[('l1', word_dtype),
                                            ('l2', word_dtype)])

    for pen in penalties:
        for r_ix, reg in enumerate(regs):
            vectors[pen]['model_regularization_param'][r_ix] = reg
            word_coefs = read_word_coefs(words, out_filename(pen, reg))
            for word in words:
                vectors[pen][word][r_ix] = word_coefs.get(word, 0)

    # Calculate the effective L1 penality for each regularization value
    # and normalize, such that each weight is represented as a
    # percentage of model penalty. The L1 and L2 penalties co-vary
    # monotonically, so this will distort proportions, but will not
    # affect the order of the results.
    for pen in penalties:
        pen_sum = np_sum(np_abs(vectors[pen][word]) for word in words)
        for word in words:
            vectors[pen][word] /= pen_sum

    return vectors

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

class FloatRange(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __contains__(self, x):
        try:
            x = float(x)
        except ValueError:
            return False
        return self.low < x <= self.high

model_dispatch = {
    'full': generic_model,
    'quarters': quarters,
    'nations': nations,
    'gender': gender,
    'canon': canon,
    'halves': halves,
    'grid': grid,
}

allowable = set(model_dispatch)
allowable_penalty = {"l1", "l2"}
allowable_regularization = FloatRange(0, 10)

def instructions():
    print("Your options are: ")
    print()
    print("full -- model the full 700-volume dataset using default settings")
    print("quarters -- create four quarter-century models")
    print("nations -- distinguish American and British poets")
    print()

def penalty_instructions():
    print("You may optionally specify a regularization penalty type.")
    print("Your penalty options are:")
    print()
    print("l1 -- sum of absolute values (a.k.a. 'manhattan norm')")
    print("l2 -- square root of sum of squares (a.k.a. 'euclidean norm')")
    print()
    print("'l1' will penalize all values equally. 'l2' will relax the")
    print("penalty when weights are spread more evenly across values.")
    print("The default is 'l2'")
    print()

def regularization_instructions():
    print("You may optionally specify a regularization parameter. (C)")
    print("Lower values make the model more tolerant of outliers.")
    print("Higher values make the model fit the training data more")
    print("closely, but may cause the model to perform poorly on")
    print("new data. (In other words, training error will go down,")
    print("but test error may go up.)")
    print()
    print("You may select a value between 0.0 and 10.0. Default is")
    print("0.0007")
    print()

def get_args():
    args = dict(zip(['_', 'command', 'penalty', 'regularization'], sys.argv))
    command = args.get('command')
    penalty = args.get('penalty', 'l2')
    regularization = args.get('regularization', 0.0007)
    if (command not in allowable or
            penalty not in allowable_penalty or
            regularization not in allowable_regularization):
        instructions()
        sys.exit(0)
    return command, penalty, float(regularization)

def get_clui_args():
    inst = instructions, penalty_instructions, regularization_instructions
    allow = allowable, allowable_penalty, allowable_regularization
    default = 'full', 'l2', 0.00007
    prompt = ("Which option do you want to run? ",
              "Which penalty would you like to use? ",
              "What parameter would you like to use? ")
    command, penalty, regularization = map(
        get_clui_input, inst, allow, default, prompt
    )

    return command, penalty, float(regularization)

def get_clui_input(instructions, allow, default, prompt):
    instructions()
    result = ""
    while result not in allow:
        result = input(prompt)
        if result == "":
            result = default
    print()
    return result

def run_model(model_dispatch):
    if len(sys.argv) > 1:
        command, penalty, regularization = get_args()
    else:
        command, penalty, regularization = get_clui_args()

    s = Settings()
    s.penalty = penalty
    s.regularization = regularization
    model_dispatch[command](s)

if __name__ == '__main__':
    run_model(model_dispatch)
