import parallel_crossvalidate as pc
import cheatsheet
import sys

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
    # We don't ordinarily include canonical volumes that were not in either
    # sample.
    excludeif = {'pubname': 'TEM',
                 'recept': 'addcanon'}

    excludeifnot = {}
    excludebelow = {'firstpub': 1700}
    excludeabove = {'firstpub': 1950}
    sizecap = 360

    # THRESHOLDS
    # For more historically-interesting kinds of questions, we limit the part
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

    # VALIDATION SETTINGS
    # kfold_step = 4
    kfold_step = 1

    # GRID SEARCH SETTINGS
    start_exp = 1
    end_exp = -2
    granularity = 3
    selection_threshold = 0.001

    @property
    def exclusions(self):
        return (self.excludeif, self.excludeifnot, self.excludebelow,
                self.excludeabove, self.sizecap)

def model_training_data(settings):
    volumes = pc.VolumeMeta(settings.sourcefolder, settings.extension,
                            settings.classpath, settings.exclusions,
                            settings.category2sorton, settings.positive_class)
    training = pc.TrainingData(volumes, settings.pastthreshold,
                               settings.futurethreshold, settings.datetype,
                               settings.numfeatures, settings.kfold_step)
    return training

def model_output(model, path, verbose=False):
    model.write_output_rows(path)
    model.write_coefficients(path)
    if verbose:
        model.display_coefficients()

    tiltaccuracy = pc.diachronic_tilt(model.allvolumes, [])
    pc.display_tilt_accuracy(model.accuracy(), tiltaccuracy)

def leave_one_out_model(settings):
    training = model_training_data(settings)
    model = pc.LeaveOneOutModel(
        training, settings.penalty, settings.regularization
    )

    model_output(model, settings.outputpath)

def nations(settings):
    settings.outputpath = 'nationalpredictions.csv'
    settings.pastthreshold = 1700
    settings.positive_class = 'uk'
    settings.category2sorton = 'nation'
    leave_one_out_model(settings)

def gender(settings):
    settings.outputpath = 'genderpredictions.csv'
    settings.pastthreshold = 1700
    settings.positive_class = 'f'
    settings.category2sorton = 'gender'
    leave_one_out_model(settings)

def canon(settings):
    settings.outputpath = 'canonpredictions.csv'
    del settings.excludeif['recept']
    settings.sizecap = 450
    leave_one_out_model(settings)

def halves(settings):
    # DO THE FIRST HALF.
    settings.outputpath = 'firsthalfpredictions.csv'
    settings.excludebelow['firstpub'] = 1800
    settings.excludeabove['firstpub'] = 1875
    settings.sizecap = 300
    leave_one_out_model(settings)

    # NOW DO THE SECOND HALF.
    settings.outputpath = 'secondhalfpredictions.csv'
    settings.excludebelow['firstpub'] = 1876
    settings.excludeabove['firstpub'] = 1925
    leave_one_out_model(settings)

def quarters(settings):
    quarteroptions = [('1820-44predictions.csv', 1800, 1844),
                      ('1845-69predictions.csv', 1845, 1869),
                      ('1870-94predictions.csv', 1870, 1894),
                      ('1895-19predictions.csv', 1895, 1925)]
    # I removed the `quarterresults` line here; it seemed to do nothing. --SE

    for outputpath, pastthreshold, futurethreshold in quarteroptions:
        print(pastthreshold)
        settings.outputpath = outputpath
        settings.pastthreshold = pastthreshold
        settings.futurethreshold = futurethreshold

        leave_one_out_model(settings)
        # I removed the `theseresults` lines here; see above. --SE

def grid(settings):
    training = model_training_data(settings)
    grid = pc.GridSearch(
        training, settings.start_exp, settings.end_exp, settings.granularity,
        settings.selection_threshold
    )

    model = pc.FeatureSelectModel(
        training, settings.penalty, settings.regularization,
        feature_selector=grid
    )

    model_output(model, 'gridfinalmodel.csv')

def gridcheat(settings):
    training = model_training_data(settings)
    words = cheatsheet.words
    training.set_vocablist(words)
    model = pc.LeaveOneOutModel(training, settings.penalty,
                                settings.regularization)
    model_output(model, 'cheatingmodel.csv')

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
    'full': leave_one_out_model,
    'quarters': quarters,
    'nations': nations,
    'gender': gender,
    'canon': canon,
    'halves': halves,
    'grid': grid,
    'gridcheat': gridcheat,
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
    print("0.000007")
    print()

def get_args():
    args = dict(zip(['_', 'command', 'penalty', 'regularization'], sys.argv))
    command = args.get('command')
    penalty = args.get('penalty', 'l2')
    regularization = args.get('regularization', 0.00007)
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

    settings = Settings()
    settings.penalty = penalty
    settings.regularization = regularization
    model_dispatch[command](settings)

if __name__ == '__main__':
    run_model(model_dispatch)
