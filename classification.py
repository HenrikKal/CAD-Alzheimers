from random import random

import dicom
import os
import numpy as np
from matplotlib import pyplot
import nibabel as nib
from nilearn import plotting
from sklearn import svm
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import pywt
# tutorial credits
# https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB




def plot_pet(path):
    dcm_files = []
    for dir_name, sub_dir, files in os.walk(path):
        for file in files:
            if ".dcm" in file.lower():
                dcm_files.append(path + file)



    ref = dicom.read_file(dcm_files[0])

    pix_dim = (int(ref.Rows), int(ref.Columns), len(dcm_files))

    pix_spacing = (float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1]), float(ref.SliceThickness))

    x = np.arange(0.0, (pix_dim[0]+1)*pix_spacing[0], pix_spacing[0])
    y = np.arange(0.0, (pix_dim[1]+1)*pix_spacing[1], pix_spacing[1])
    z = np.arange(0.0, (pix_dim[2]+1)*pix_spacing[2], pix_spacing[2])

    array_dicom = np.zeros(pix_dim, dtype=ref.pixel_array.dtype)

    for file_name in dcm_files:
        ds = dicom.read_file(file_name)
        array_dicom[:, :, dcm_files.index(file_name)] = ds.pixel_array

    for row in array_dicom[:, :, 80]:
        print(row)

    pyplot.imshow(array_dicom[:, :, 80])



    pyplot.show()



def plot_mri(images, nr):
    data = images[nr-1].get_data()


   # plotting.plot_anat(mri_path, display_mode='z', cut_coords=[5])
    #plotting.show()

    pyplot.imshow(data[60, :, :])
    pyplot.show()



def read_pet_images(path):
    pet_images = []
    for dir_name, sub_dir, files in os.walk(path):
        for sub in sub_dir:

            for d, s, files in os.walk(path+"/"+sub+"/"):
                    dcm_files = []
                    for file in files:
                        if ".dcm" in file.lower():
                            dcm_files.append(path + "/" + sub + "/" + "/" + file)


                    ref = dicom.read_file(dcm_files[0])
                    pix_dim = (int(ref.Rows), int(ref.Columns), len(dcm_files))

                    array_dicom = np.zeros(pix_dim, dtype=ref.pixel_array.dtype)

                    for file_name in dcm_files:
                        ds = dicom.read_file(file_name)
                        array_dicom[:, :, dcm_files.index(file_name)] = ds.pixel_array

                    matrix = array_dicom[:, :, 88]
                    pet_images.append(matrix)



    return pet_images

def read_mri_images(path):
    images = []
    for dir_name, sub_dir, files in os.walk(path):
        for file in files:
            if ".nii" in file.lower():
                img = nib.load(path+file)
                images.append(img.get_data())

    return images





def apply_dwt(brains):
    coeffs = []
    i = 1
    for brain in brains:
        #print(len(brains))
       # #print(brain.shape[0])
        #print(brain.shape[2])
        #matrix = brain[0]
        #print(matrix.shape)
        col = 0
        #print(matrix[0:brain.shape[0], 1440:1600])
        average_matrix = brain[0]
        for image in range(0,len(brain)):
                #print("m:", matrix[0:brain.shape[0], col:col+brain.shape[2]])
                #print("c:", col)
                #matrix[0:brain.shape[0], col:col+brain.shape[2]] = brain[image]
               # print("0-", brain.shape[0], "&", str(col)+"-"+str(col+brain.shape[2]))
                #print(brain[image].shape)
                #col += brain.shape[2]
                #matrix = np.concatenate((matrix,brain[image]), axis=1)
                average_matrix += brain[image]
       # print(matrix.shape)
        #print("lul",i, matrix[90][660:690])
        #i+=1

        average_matrix /= len(brain)
        #print(average_matrix)
        p = pywt.wavedec2(average_matrix, 'haar', level=3)[1][2]
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
    pca = PCA(n_components=None)
    pca.fit(matrix)
    pca_matrix = pca.transform(matrix)


    return pca_matrix




def matrix_to_vector(matrix):
    # unfold matrix into one vector
    vector = np.zeros(len(matrix)*len(matrix[0]))
    # row in matrix
    for row_nr in range(matrix.shape[0]):

        row = matrix[row_nr]
        # element in row
        for el in range(len(row)):
            vector[row_nr*el] = matrix[row_nr][el]


    return vector


def print_matrix(matrix):

    for row in matrix:
        print(row)

#pet_ad = read_pet_images("C:/Users/Henrik/Desktop/PET_AD/")
#pet_normal = read_pet_images("C:/Users/Henrik/Desktop/PET_NORMAL/")
mri_ad = read_mri_images("C:/Users/Henrik/Desktop/ad/")
mri_normal = read_mri_images("C:/Users/Henrik/Desktop/normal/")

mri_ad_192 = []
mri_ad_256 = []
for im in mri_ad:
    if im.shape == (192,192,160):
        mri_ad_192.append(im)
    elif im.shape == (256,256,166):
        mri_ad_256.append(im)

mri_normal_192 = []
mri_normal_256 = []
for im in mri_normal:
    if im.shape == (192,192, 160):
        mri_normal_192.append(im)
    elif im.shape == (256, 256, 166):
        mri_normal_256.append(im)


img_192 = mri_ad_192+mri_normal_192
img_256 = mri_ad_256+mri_normal_256
images = mri_ad + mri_normal
#images = pet_ad + pet_normal

targets_192 = []
targets_256 = []
targets_192 += ["AD"]*len(mri_ad_192)
targets_192 += ["NL"]*len(mri_normal_192)
targets_256 += ["AD"]*len(mri_ad_256)
targets_256 += ["NL"]*len(mri_normal_256)

#targets += ["AD"]*len(pet_ad)
#targets += ["NL"]*len(pet_normal)
y = targets_192
X = img_192

#X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.4, random_state=2)

loo = LeaveOneOut()
svmArray = []
rfArray = []
nbArray = []
for train_index, test_index in loo.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train = [X[i] for i in train_index]
    X_test = [X[i] for i in test_index]
    y_train = [y[i] for i in train_index]
    y_test = [y[i] for i in test_index]

    #y_train, y_test = y[train_index], y[test_index]
    #print(X_train, X_test, y_train, y_test)


    coeffs_all = apply_dwt(X_train+X_test)
    coeffs_train = apply_dwt(X_train)
    coeffs_test = apply_dwt(X_test)
    #print("all\n")
    matrix_all = create_matrix(coeffs_all)
    #print("train\n")
    matrix_train = create_matrix(coeffs_train)
    #print("test\n")
    matrix_test = create_matrix(coeffs_test)

    pca_train = apply_pca(matrix_train)
    pca_train = pca_train.transpose()

    pca_test = apply_pca(matrix_test)
    pca_test = pca_test.transpose()


    clf = svm.SVC(kernel='linear')
    clf.fit(pca_train, y_train)


    #print("SVM")
    #print(clf.score(pca_test, y_test))
    #print(clf.predict(pca_test))
    #print(y_test)
    svmArray.append(clf.score(pca_test, y_test))

    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
    rf.fit(pca_train, y_train)

    #print("\nRANDOM FOREST")
    #print(rf.score(pca_test, y_test))
    #print(rf.predict(pca_test))
    #print(y_test)
    rfArray.append(rf.score(pca_test, y_test))

    #print("\nNAIVE BAYES")
    gnb = BernoulliNB()
    gnb.fit(pca_train, y_train)

    """print(gnb.score(pca_test, y_test))
    print(gnb.predict(pca_test))
    print(y_test)"""
    nbArray.append(gnb.score(pca_test, y_test))
print("------------------------------LOO---------------------------")
print("svmAvg:")
print(np.mean(svmArray))
print("rfMean:")
print(np.mean(rfArray))
print("nbMean:")
print(np.mean(nbArray))

# 3 fold:
svmArray = []
rfArray = []
nbArray = []
kf = KFold(n_splits=3)
KFold(n_splits=3, random_state=0, shuffle=False)
for train_index, test_index in kf.split(X):

    X_train = [X[i] for i in train_index]
    X_test = [X[i] for i in test_index]
    y_train = [y[i] for i in train_index]
    y_test = [y[i] for i in test_index]

    # y_train, y_test = y[train_index], y[test_index]
    # print(X_train, X_test, y_train, y_test)


    coeffs_all = apply_dwt(X_train + X_test)
    coeffs_train = apply_dwt(X_train)
    coeffs_test = apply_dwt(X_test)
    # print("all\n")
    matrix_all = create_matrix(coeffs_all)
    # print("train\n")
    matrix_train = create_matrix(coeffs_train)
    # print("test\n")
    matrix_test = create_matrix(coeffs_test)

    pca_train = apply_pca(matrix_train)
    pca_train = pca_train.transpose()

    pca_test = apply_pca(matrix_test)
    pca_test = pca_test.transpose()

    clf = svm.SVC(kernel='linear')
    clf.fit(pca_train, y_train)

    # print("SVM")
    # print(clf.score(pca_test, y_test))
    # print(clf.predict(pca_test))
    # print(y_test)
    svmArray.append(clf.score(pca_test, y_test))

    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
    rf.fit(pca_train, y_train)

    # print("\nRANDOM FOREST")
    # print(rf.score(pca_test, y_test))
    # print(rf.predict(pca_test))
    # print(y_test)
    rfArray.append(rf.score(pca_test, y_test))

    # print("\nNAIVE BAYES")
    gnb = BernoulliNB()
    gnb.fit(pca_train, y_train)

    """print(gnb.score(pca_test, y_test))
    print(gnb.predict(pca_test))
    print(y_test)"""
    nbArray.append(gnb.score(pca_test, y_test))
print("------------------------------3-Fold---------------------------")
print("svmAvg:")
print(np.mean(svmArray))
print("rfMean:")
print(np.mean(rfArray))
print("nbMean:")
print(np.mean(nbArray))





# I'm following this
#https://www.youtube.com/watch?v=_lY74pXWlS8

# So currently i'm applying DWT to each image individualyl, then i'm converting the matrix that
# I get from it into a vector simply by unfolding the matrix. For each image I get such a vector.
# Right now i'm trying to combine these vectors into a matrix, by inserting them as columns.
# Current problem is inserting the vector columns.
# Eventually, the plan is to run PCA on this entire matrix.