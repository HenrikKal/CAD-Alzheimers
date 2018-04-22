from imageutils import read_pet_images, read_mri_images
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import svm
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing
import pywt
from matplotlib import pyplot
import numpy as np
from datetime import datetime





#############################
### PROCESSING
#############################


def apply_dwt(brains, lvl):
    coeffs = []
    for brain in brains:

        average_matrix = brain[59]

      #  for i in range(0, brain.shape[2]):
            #print(brain[image])

           # average_matrix += brain[i]

        #average_matrix = np.true_divide(average_matrix, brain.shape[2])
        p = pywt.wavedec2(average_matrix, 'haar', level=lvl)[1][2]
        p = p.astype(float)
        p = preprocessing.scale(p)
        coeffs.append(p)



    return coeffs



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




def matrix_to_vector(matrix):
    # unfold matrix into one vector
    vector = np.zeros(len(matrix)*len(matrix[0]))
    # row in matrix
    for row_nr in range(len(matrix[0])):
        row = matrix[row_nr]
        # element in row
        for el in range(len(row)):
            vector[row_nr*el] = matrix[row_nr][el]


    return vector




def process_data(X_train, X_test, settings):
    use_pca = settings[3]
    dwt_lvl = settings[4]

    dwt_train = apply_dwt(X_train, dwt_lvl)
    dwt_test = apply_dwt(X_test, dwt_lvl)
    matrix_train = create_matrix(dwt_train)
    matrix_test = create_matrix(dwt_test)

    if use_pca:
        matrix_train = apply_pca(matrix_train)
        matrix_test = apply_pca(matrix_test)

    return matrix_train.transpose(), matrix_test.transpose()


####################################
### TRAINING
###################################


def train_models(matrix_train, y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(matrix_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, oob_score=True)
    rf.fit(matrix_train, y_train)

    gnb = BernoulliNB()
    gnb.fit(matrix_train, y_train)

    return clf, rf, gnb




def predict(classifier, matrix_test, y_test):
    score = classifier.score(matrix_test, y_test)
    predictions = classifier.predict(matrix_test)
    results = (score, predictions, y_test)

    return results






def split_data(ad_images, normal_images, test_s):
    targets = []

    targets += ["AD"] * len(ad_images)
    targets += ["NL"] * len(normal_images)
    X_train, X_test, y_train, y_test = train_test_split(ad_images+normal_images, targets, test_size=test_s)

    return X_train, X_test, y_train, y_test





def test_accuracy(pet_ad, pet_normal, settings):
    svm_array = []
    rf_array = []
    nb_array = []

    for i in range(1, settings[2]+1):
        print("ITERATION: ", i, "/", settings[2])
        X_train,  X_test, y_train, y_test = split_data(pet_ad, pet_normal, 0.4)
        matrix_train, matrix_test = process_data(X_train, X_test, settings)
        svm, rf, nb = train_models(matrix_train, y_train)
        svm_array.append(predict(svm, matrix_test, y_test)[0])
        rf_array.append(predict(rf, matrix_test, y_test)[0])
        nb_array.append(predict(nb, matrix_test, y_test)[0])


    return np.mean(svm_array), np.mean(rf_array),  np.mean(nb_array)


def write_test_log(settings, results):


    log_string = str(datetime.now()) + "\n"
    log_string += "nr_AD_images: " + str(settings[0]) + "\n"
    log_string += "nr_Normal_images: " + str(settings[1]) + "\n"
    log_string += "PCA applied: " + str(settings[3]) + "\n"
    log_string += "DWT level: " + str(settings[4]) + "\n"
    log_string += "nr_iterations: " + str(settings[2]) + "\n"
    log_string += "SVM acc: " + str(results[0]) + "\n"
    log_string += "RF acc: " + str(results[1]) + "\n"
    log_string += "NB acc: " + str(results[2]) + "\n"
    log_string += "\n"

    with open("C:/Users/Henrik/Desktop/test_log.txt", "a") as log_file:
        log_file.write(log_string)




def main(settings):
    print("START READING")
    pet_ad = read_pet_images("C:/Users/Henrik/Desktop/PET_AD_CLEAN/", settings[0])
    pet_normal = read_pet_images("C:/Users/Henrik/Desktop/PET_NORMAL_CLEAN/", settings[1])
    print("DONE READING")



    results = test_accuracy(pet_ad, pet_normal, settings)
    print("SVM")
    print(results[0])
    print("RF")
    print(results[1])
    print("NB")
    print(results[2])

    write_test_log(settings, results)


# settings describes how data is processed,
# format: (nr_ad, nr_normal, use_pca, dwt_lvl)
nr_ad = 25
nr_normal = 25
iterations = 200
use_pca = False
dwt_lvl = 3

settings = (nr_ad, nr_normal, iterations, use_pca, dwt_lvl)


main(settings)




