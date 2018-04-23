from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing
import pywt
import numpy as np
from cad_log import write_test_log
from multiprocessing import Pool, Process




#############################
### PROCESSING
#############################



def apply_dwt(brains, lvl, slices):
    coeffs = []
    for brain in brains:
        start_slice = len(brain)//2-3
        end_slice = start_slice+5

        matrix = brain[start_slice]


        for i in range(start_slice+1, end_slice+1):
            matrix = np.concatenate((matrix, brain[i]), axis=1)


        p = pywt.wavedec2(matrix, 'haar', level=lvl)[0]
        p = p.astype(float)
        p = preprocessing.scale(p)
        coeffs.append(p)

    return coeffs



# Creates a single matrix from all the matrices retrieved from DWT by
# transforming each matrix into a column vector, and then appending each column
def create_matrix(coeffs):

    vector = matrix_to_vector(coeffs[0])
    matrix = np.zeros((len(vector), 0))
    matrix = np.insert(matrix, 0, vector, 1)

    for i in range(1, len(coeffs)):
        v = matrix_to_vector(coeffs[i])
        matrix = np.insert(matrix, i, v, 1)

    return matrix



def apply_pca(matrix):
    pca = PCA(n_components=6)
    matrix = matrix.transpose()
    pca.fit(matrix)
    pca_matrix = pca.transform(matrix)
    return (pca_matrix)


# Unfolds matrix into one vector
def matrix_to_vector(matrix):
    vector = np.zeros(len(matrix)*len(matrix[0]))
    # row in matrix
    for row_nr in range(matrix.shape[0]):
        row = matrix[row_nr]
        # element in row
        for el in range(len(row)):
            vector[row_nr*el] = matrix[row_nr][el]


    return vector



# Processes both the training data and the test input data
# according to the specified settings.
def process_data(X_train, X_test, settings):
    use_pca = settings[3]
    dwt_lvl = settings[4]
    slices = settings[5]

    # DWT for each image
    dwt_train = apply_dwt(X_train, dwt_lvl, slices)
    dwt_test = apply_dwt(X_test, dwt_lvl, slices)

    # Create single matrix
    matrix_train = create_matrix(dwt_train)
    matrix_test = create_matrix(dwt_test)

    if use_pca:
        matrix_train = apply_pca(matrix_train)
        matrix_test = apply_pca(matrix_test)
    else:
        matrix_train = matrix_train.transpose()
        matrix_test = matrix_test.transpose()

    return matrix_train, matrix_test


####################################
### TRAINING
###################################

# Sets up the models
def train_models(matrix_train, y_train):
    clf = svm.SVC(kernel='rbf', degree=6, C=6.0)
    clf.fit(matrix_train, y_train)

    rf = RandomForestClassifier(n_estimators=80, oob_score=True)
    rf.fit(matrix_train, y_train)

    gnb = GaussianNB()
    gnb.fit(matrix_train, y_train)

    return clf, rf, gnb



# Specified classifier predicts input test data
# and compares it to expected output
# Returns accuracy, the actual predictions and the expected output
def predict(classifier, matrix_test, y_test):
    score = classifier.score(matrix_test, y_test)
    predictions = classifier.predict(matrix_test)
    results = (score, predictions, y_test)

    return results




# Takes the ad and normal images and splits them
# according to specified test size
def split_data(ad_images, normal_images, test_s):
    targets = []

    targets += ["AD"] * len(ad_images)
    targets += ["NL"] * len(normal_images)
    X_train, X_test, y_train, y_test = train_test_split(ad_images+normal_images, targets, test_size=test_s)

    return X_train, X_test, y_train, y_test




# Trains and tests the models for specified number of iterations
# Returns the mean accuracy for all models
def test_accuracy(pet_ad, pet_normal, settings):
    svm_array = []
    rf_array = []
    nb_array = []
    iter = settings[2]
    test_size = settings[6]

    for i in range(1, iter+1):
        print("ITERATION: ", i, "/", iter)

        # Split data
        X_train,  X_test, y_train, y_test = split_data(pet_ad, pet_normal, test_size)

        # Process data
        matrix_train, matrix_test = process_data(X_train, X_test, settings)

        # Init models
        svm, rf, nb = train_models(matrix_train, y_train)

        # Test
        svm_acc = predict(svm, matrix_test, y_test)[0]
        rf_acc = predict(rf, matrix_test, y_test)[0]
        nb_acc = predict(nb, matrix_test, y_test)[0]


        svm_array.append(svm_acc)
        rf_array.append(rf_acc)
        nb_array.append(nb_acc)


    return np.mean(svm_array), np.mean(rf_array),  np.mean(nb_array)






def main(cache, settings):


    pet_ad = cache[0]
    pet_normal = cache[1]

    svm_acc, rf_acc, nb_acc = test_accuracy(pet_ad, pet_normal, settings)
    print("SVM")
    print(svm_acc)
    print("RF")
    print(rf_acc)
    print("NB")
    print(nb_acc)

    write_test_log("C:/Users/Henrik/Desktop/test_log.txt", settings, (svm_acc, rf_acc, nb_acc))





# If you change something, hit CTRL + S in this file before reloading
def run(cache, nr_ad, nr_normal):
    # settings describes how data is processed,
    # format: (nr_ad, nr_normal, use_pca, dwt_lvl)
    iterations = 1000
    use_pca = False
    dwt_lvl = 4
    slices = [48]
    test_size = 0.3

    print("iterations:", iterations)
    print("Use PCA: ", use_pca)
    print("DWT lvl: ", dwt_lvl)
    print("test_size: ", test_size)


    settings = (nr_ad, nr_normal, iterations, use_pca, dwt_lvl, slices, test_size)

    print("RUN")
    main(cache, settings)



