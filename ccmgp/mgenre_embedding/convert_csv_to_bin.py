import os

import ccmgp.utils.utils as utils


def get_files(dir_path):
    """ Enumerate files from a folder recursively"""
    files = os.listdir(dir_path)
    all_files = list()
    for entry in files:
        # Create full path
        full_path = os.path.join(dir_path, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_files(full_path)
        else:
            all_files.append(full_path)
    return all_files

# Convert all the generated csv files in binary files
for folder in [utils.COMP_EMB_DIR, utils.LASER_EMB_DIR, utils.TRANSFORM_EMB_DIR, utils.RETRO_EMB_DIR]:
    files = get_files(folder)
    for f in files:
        if f.endswith('.csv'):
            utils.csv_to_binary(f)
            os.remove(f)
