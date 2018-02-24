import dicom
import os
import numpy
from matplotlib import pyplot

# tutorial credits
# https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/

path = 'C:/Users/Henrik/Downloads/PET_scans/ADNI/941_S_5193/AV45_Coreg,_Avg,_Standardized_Image_and_Voxel_Size/2015-06-10_14_30_00.0/I500613/'
dcm_files = []
for dir_name, sub_dir, files in os.walk(path):
    for file in files:
        if ".dcm" in file.lower():
            dcm_files.append(path + file)



ref = dicom.read_file(dcm_files[0])

pix_dim = (int(ref.Rows), int(ref.Columns), len(dcm_files))

pix_spacing = (float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1]), float(ref.SliceThickness))

x = numpy.arange(0.0, (pix_dim[0]+1)*pix_spacing[0], pix_spacing[0])
y = numpy.arange(0.0, (pix_dim[1]+1)*pix_spacing[1], pix_spacing[1])
z = numpy.arange(0.0, (pix_dim[2]+1)*pix_spacing[2], pix_spacing[2])

array_dicom = numpy.zeros(pix_dim, dtype=ref.pixel_array.dtype)

for file_name in dcm_files:
    ds = dicom.read_file(file_name)
    array_dicom[:, :, dcm_files.index(file_name)] = ds.pixel_array

pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(x, y, numpy.flipud(array_dicom[:, :, 80]))

pyplot.show()