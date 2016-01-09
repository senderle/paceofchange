# My goal in this script is to package up the modeling and
# evaluation processes we actually ran to produce the article,
# so they can be replicated without a lot of fuss.

import parallel_crossvalidate as pc
import sys

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

allowable = {"full", "quarters", "nations", "genders", "canon", "halves"}
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

def get_clui_input(instructions, allow, default, prompt):
    instructions()
    result = ""
    while result not in allow:
        result = input(prompt)
        if result == "":
            result = default
    print()
    return result

def display_tilt_accuracy(rawaccuracy, tiltaccuracy):
    msg = 'If we divide the dataset with a horizontal line at 0.5, accuracy is:'
    print(msg, str(rawaccuracy))
    msg = "Divided with a line fit to the data trend, it's"
    print(msg, str(tiltaccuracy))

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
        return self.pastthreshold, s.futurethreshold

    @property
    def classifyconditions(self):
        return (self.category2sorton, self.positive_class, self.datetype,
                self.numfeatures, self.regularization, self.penalty)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command, penalty, regularization = get_args()
    else:
        command, penalty, regularization = get_clui_args()

    s = Settings()
    s.penalty = penalty
    s.regularization = regularization

    if command == 'full':
        rawaccuracy, allvolumes, coefficientuples = pc.create_model(
            s.paths, s.exclusions, s.thresholds, s.classifyconditions
        )

        tiltaccuracy = pc.diachronic_tilt(allvolumes, 'linear', [])
        display_tilt_accuracy(rawaccuracy, tiltaccuracy)

    elif command == 'quarters':
        quarteroptions = [('1820-44predictions.csv', 1800, 1844),
                          ('1845-69predictions.csv', 1845, 1869),
                          ('1870-94predictions.csv', 1870, 1894),
                          ('1895-19predictions.csv', 1895, 1925)]

        quarterresults = list()

        for outputpath, pastthreshold, futurethreshold in quarteroptions:

            print(pastthreshold)
            s.outputpath = outputpath
            s.pastthreshold = pastthreshold
            s.futurethreshold = futurethreshold

            rawaccuracy, allvolumes, coefficientuples = pc.create_model(
                s.paths, s.exclusions, s.thresholds, s.classifyconditions
            )

            tiltaccuracy = pc.diachronic_tilt(allvolumes, 'linear', [])
            display_tilt_accuracy(rawaccuracy, tiltaccuracy)

            theseresults = dict()
            theseresults['allvolumes'] = allvolumes
            theseresults['rawaccuracy'] = rawaccuracy
            theseresults['tiltaccuracy'] = tiltaccuracy
            theseresults['coefficientuples'] = coefficientuples
            theseresults['startdate'] = pastthreshold

            quarterresults.append(theseresults)

    elif command == 'nations':
        s.outputpath = 'nationalpredictions.csv'
        s.pastthreshold = 1700
        s.positive_class = 'uk'
        s.category2sorton = 'nation'
        numfeatures = 3200

        rawaccuracy, allvolumes, coefficientuples = pc.create_model(
            s.paths, s.exclusions, s.thresholds, s.classifyconditions
        )

        tiltaccuracy = pc.diachronic_tilt(allvolumes, 'linear', [])
        display_tilt_accuracy(rawaccuracy, tiltaccuracy)

    elif command == 'genders':
        s.outputpath = 'genderpredictions.csv'
        s.pastthreshold = 1700
        s.positive_class = 'f'
        s.category2sorton = 'gender'

        rawaccuracy, allvolumes, coefficientuples = pc.create_model(
            s.paths, s.exclusions, s.thresholds, s.classifyconditions
        )

        tiltaccuracy = pc.diachronic_tilt(allvolumes, 'linear', [])
        display_tilt_accuracy(rawaccuracy, tiltaccuracy)

    elif command == 'canon':
        s.outputpath = 'canonpredictions.csv'
        del s.excludeif['recept']
        s.sizecap = 450

        rawaccuracy, allvolumes, coefficientuples = pc.create_model(
            s.paths, s.exclusions, s.thresholds, s.classifyconditions
        )

        tiltaccuracy = pc.diachronic_tilt(allvolumes, 'linear', [])
        display_tilt_accuracy(rawaccuracy, tiltaccuracy)

    elif command == 'halves':
        s.outputpath = 'firsthalfpredictions.csv'
        s.excludebelow['firstpub'] = 1800
        s.excludeabove['firstpub'] = 1875
        s.sizecap = 300

        rawaccuracy, allvolumes, coefficientuples = pc.create_model(
            s.paths, s.exclusions, s.thresholds, s.classifyconditions
        )

        tiltaccuracy = pc.diachronic_tilt(allvolumes, 'linear', [])
        display_tilt_accuracy(rawaccuracy, tiltaccuracy)

        # NOW DO THE SECOND HALF.

        s.excludebelow['firstpub'] = 1876
        s.excludeabove['firstpub'] = 1925
        s.outputpath = 'secondhalfpredictions.csv'

        rawaccuracy, allvolumes, coefficientuples = pc.create_model(
            s.paths, s.exclusions, s.thresholds, s.classifyconditions
        )

        tiltaccuracy = pc.diachronic_tilt(allvolumes, 'linear', [])
        display_tilt_accuracy(rawaccuracy, tiltaccuracy)
