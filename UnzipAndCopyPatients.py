import os
import shutil

"""
Usage:
     Replace the path variable below to point to the folder containing all the
     patient data (the containing all the different University folders)

     Replace the dest variable to point to where the .nii files of each patient
     should be stored.

     This script does a DFS traversal of the directory pointed to by "path", and
     copies the .gz files it finds at the leaf nodes. These files are written to
     the directory pointed to by "dest", and are unzipped.
"""
path = '/Users/Tim/Desktop/Code/Machine_Learning/TZO/NYU/Thingy'
dest = '/Users/Tim/Desktop/Code/Machine_Learning/TZO/NYU/Patients/'

rootDir = path
for dirName, subdirList, fileList in os.walk(rootDir, topdown=True):
    header = '_'.join(dirName[55:].split('\\'))[1::]
    for fname in fileList:
        shutil.copy(dirName + "/" + fname, dest + header + '_' + fname)
        print(os.system("gzip -d ../Patients/" + header + '_' + fname))
