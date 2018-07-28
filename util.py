import os
import shutil
import math

cwd = os.getcwd()
print(cwd)


def remove_folder_content(dir_name):
    shutil.rmtree(os.path.join(cwd, dir_name))


def split_train_val():
    val_folder_normal = os.path.join(cwd, 'chest_xray/val/NORMAL')
    val_folder_pneumonia = os.path.join(cwd, 'chest_xray/val/PNEUMONIA')

    if not os.path.isdir(val_folder_normal):
        os.makedirs(val_folder_normal)

    if not os.path.isdir(val_folder_pneumonia):
        os.makedirs(val_folder_pneumonia)

    train_dirs = ['chest_xray/train/NORMAL', 'chest_xray/train/PNEUMONIA', val_folder_normal, val_folder_pneumonia]

    for dir in train_dirs:
        files = [os.path.join(dir, f) for f in os.listdir(dir)]
        total_files = len(files)
        N = math.ceil(total_files / 5)
        i = 0
        print(total_files)
        if dir == 'chest_xray/train/NORMAL':
            dir_move_to = val_folder_normal
        else:
            dir_move_to = val_folder_pneumonia

        for fi in files:
            if i < N:
                f_base = os.path.basename(fi)
                shutil.move(fi, os.path.join(dir_move_to, f_base))
                print ('move {0}, name {1}'.format(i, f_base))
            i += 1


def move_files():
    """Move files into subdirectories."""

    """split images in train folder into train / val"""
    split_train_val()

    bacteria = 'BACTERIA'
    virus = 'VIRUS'
    sections = ['train', 'test', 'val']
    for section in sections:
        bacteria_subFolder = os.path.join(cwd, 'chest_xray/{0}/PNEUMONIA_BACTERIA'.format(section))
        viral_subFolder = os.path.join(cwd, 'chest_xray/{0}/PNEUMONIA_VIRAL'.format(section))

        if not os.path.isdir(bacteria_subFolder):
            os.makedirs(bacteria_subFolder)

        if not os.path.isdir(viral_subFolder):
            os.makedirs(viral_subFolder)

        dir = 'chest_xray/{0}/PNEUMONIA'.format(section)

        files = [os.path.join(dir, f) for f in os.listdir(dir)]

        for fi in files:
            f_base = os.path.basename(fi)
            if f_base.startswith(bacteria):
                dir_move_to = bacteria_subFolder
            elif f_base.startswith(virus):
                dir_move_to = viral_subFolder

            shutil.move(fi, os.path.join(dir_move_to, f_base))
            print ('move {0}, to folder {1}'.format(f_base, dir_move_to))

        files = [os.path.join(dir, f) for f in os.listdir(dir)]

        if len(files) == 0:
            shutil.rmtree(dir)

