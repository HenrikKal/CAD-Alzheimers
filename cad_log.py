from datetime import datetime

#####################################################
## Write logs of test runs to the specified file path
#####################################################


def write_test_log(path, settings, results):


    log_string = str(datetime.now()) + "\n"
    log_string += "nr_AD_images: " + str(settings[0]) + "\n"
    log_string += "nr_Normal_images: " + str(settings[1]) + "\n"
    log_string += "slices: " + str(settings[5]) + "\n"
    log_string += "PCA applied: " + str(settings[3]) + "\n"
    log_string += "DWT level: " + str(settings[4]) + "\n"
    log_string += "nr_iterations: " + str(settings[2]) + "\n"
    log_string += "SVM acc: " + str(results[0]) + "\n"
    log_string += "RF acc: " + str(results[1]) + "\n"
    log_string += "NB acc: " + str(results[2]) + "\n"
    log_string += "\n"

    with open(path, "a") as log_file:
        log_file.write(log_string)
