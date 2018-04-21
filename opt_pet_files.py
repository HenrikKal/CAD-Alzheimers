from shutil import copyfile
import os


def read_folder(path, dst):
    subject_dir = []
    all_files = []
    subject_files = []


    prev = ""
    for dir_name, sub_dir, files in os.walk(path):
        if files != []:
            sdir = files[0][5:15]
            if sdir != prev:
                all_files.append([sdir, dir_name] + files)
                prev = sdir


    for file in all_files:
        for f in file[2:]:
            src = file[1] + "/" + f
            src = src.replace('\\', '/')
            folder = dst+"/"+file[0] + "/"
            sliceNr = f[91:93].strip('_') # get slice nr
            if not os.path.exists(folder):
                os.makedirs(folder)

            copyfile(src, folder+sliceNr+f)



    print("Done with", path)


pet_folder_ad = "C:/Users/Henrik/Desktop/PET_AD_56"
pet_folder_normal = "C:/Users/Henrik/Desktop/ADNI"


#read_folder(pet_folder_ad, "C:/Users/Henrik/Desktop/PET_AD_CLEAN")
read_folder(pet_folder_normal, "C:/Users/Henrik/Desktop/PET_NORMAL_CLEAN")