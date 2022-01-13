import pandas as pd
import os
import glob
from PIL import Image

# list train folder structure
def folders(path,fname):
    if fname=='seg_pred':
        files = glob.glob(pathname=path + 'seg_pred//' + '*.jpg')
        print(f'Prediction folder has: {len(files)}')
    else:
        for folder in os.listdir(path+fname):
            files = glob.glob(pathname=path+fname+'//'+folder+'/*.jpg')
            print(f'({folder}) folder has: {len(files)}')
    return

def files(path,fname):
    act_img_size = []
    if fname=='seg_pred':
        files = glob.glob(pathname=str(path + fname+'/*.jpg'))
        for file in files:
            im = Image.open(file)
            act_img_size.append(im.size)
    else:
        for folder in os.listdir(path + fname):
            files = glob.glob(pathname=path + fname + '//' + folder + '/*.jpg')
            for file in files:
                im = Image.open(file)
                act_img_size.append(im.size)
    series = pd.Series(act_img_size, name='Height x Width').value_counts()
    return series