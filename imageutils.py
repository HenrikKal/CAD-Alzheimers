import dicom
import os
import numpy as np
from matplotlib import pyplot
import nibabel as nib

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


    #pyplot.figure(dpi=300)
    #pyplot.axes().set_aspect('equal', 'datalim')
    #pyplot.set_cmap(pyplot.gray())
    #pyplot.pcolormesh(x, y, np.flipud(array_dicom[:, :, 80]))

    pyplot.show()



def plot_mri(images, nr):
    data = images[nr-1].get_data()


   # plotting.plot_anat(mri_path, display_mode='z', cut_coords=[5])
    #plotting.show()

    pyplot.imshow(data[60, :, :])
    pyplot.show()



def read_pet_images(path, nr_images):
    pet_images = []
    for dir_name, sub_dir, files in os.walk(path):
        for sub in sub_dir[:nr_images-1]:

            for d, s, files in os.walk(path+"/"+sub+"/"):
                    dcm_files = []
                    for file in files:
                        if ".dcm" in file.lower():
                            dcm_files.append(path + "/" + sub + "/" + "/" + file)



                    ref = dicom.read_file(dcm_files[0])
                    pix_dim = (int(ref.Rows), int(ref.Columns), len(dcm_files))

                    array_dicom = np.zeros(pix_dim, dtype=ref.pixel_array.dtype)

                    for file_name in dcm_files:
                        print(file_name)
                        ds = dicom.read_file(file_name)
                        array_dicom[:, :, dcm_files.index(file_name)] = ds.pixel_array

                    matrix = array_dicom[:, :, :] # {:, :, x]
                    pet_images.append(matrix)



    return pet_images

def read_mri_images(path):
    images = []
    for dir_name, sub_dir, files in os.walk(path):
        for file in files:
            if ".nii" in file.lower():
                img = nib.load(path+file)
                images.append(img.get_data()[60, :, :])

    return images



