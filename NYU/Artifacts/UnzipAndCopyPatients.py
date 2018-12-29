import os
import shutil

"""
Usage:
     Replace the path variable below to point to the folder containing all the
     patient data (the containing all the different University folders)

     This script does a DFS traversal of the directory pointed to by "path", and
     copies the .gz files it finds at the leaf nodes. These files are written to
     a directory called Patient_Scans, and are unzipped.
"""
path = '/Users/Tim/Desktop/Code/Machine_Learning/TZO/NYU/Thingy'

curr_dir = os.getcwd()
os.chdir("..")
NYU_Folder = os.getcwd()
print(NYU_Folder)
os.system("mkdir Patient_Scans")
os.chdir(curr_dir)
dest = NYU_Folder + "/Patient_Scans/"

for dirName, subdirList, fileList in os.walk(path, topdown=True):
    header = '_'.join(dirName[55:].split('\\'))[1::]
    for fname in fileList:
        os.chdir(dest)
        os.system("mkdir " + header)
        shutil.copy(dirName + "/" + fname, dest + header + "/" + fname)
        print(os.system("gzip -d " + dest + header + "/" + fname))
