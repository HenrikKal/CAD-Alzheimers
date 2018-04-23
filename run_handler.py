import imageutils
import classifier
import sys
from importlib import reload
import os




path_ad = "D:/Alzheimers/PET_AD_CLEAN"
path_normal = "D:/Alzheimers/PET_NORMAL_CLEAN/"

nr_ad = 48
nr_normal = 48


# CREDIT: https://stackoverflow.com/questions/6687660/keep-persistent-variables-in-memory-between-runs-of-python-script
# Peter Lyons Jul 14 '11
cache = None
if __name__ == "__main__":
    while True:
        if not cache:
            pet_ad = imageutils.read_pet_images(path_ad, nr_ad)
            pet_normal = imageutils.read_pet_images(path_normal, nr_normal)
            cache = (pet_ad, pet_normal)


        try:
            classifier.run(cache, nr_ad, nr_normal)

        except RuntimeError as e:
            print("Error in classifier.py")
            print(e)


        print("Press enter to re-run the script, CTRL-C to exit")
        sys.stdin.readline()

        os.remove(getattr(classifier, '__cached__', 'classifier.pyc'))
        reload(classifier)



