#%% !/usr/bin/env python3
# Initialization

Version = '01'

# Define the script description
Description = """
    This script prints figures to help centering sample
    during compression testing. From original script of
    Benjamin Voumard

    Author: Mathieu Simon
            ARTORG Center for Biomedical Engineering Research
            SITEM Insel
            University of Bern

    Date: June 2023
    """

#%% Imports
# Modules import

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import json as js
import scipy.ndimage as ndimage
from matplotlib_scalebar.scalebar import ScaleBar

#%% Functions
# Define functions

def cross(img_np, coordinate, size=5):
    img_np[int(coordinate[0]) - size:int(coordinate[0]) + size, int(coordinate[1])] = 0
    img_np[int(coordinate[0]), int(coordinate[1]) - size:int(coordinate[1]) + size] = 0
    return img_np


def big_cross(img_np, coordinate, value=0):
    img_np[:, int(coordinate[1])] = value
    img_np[int(coordinate[0]), :] = value
    return img_np


def plot_segmentation(img_gray, img_seg, title='', alpha=0.6, save_path=''):
    img_seg = img_seg.astype(int)
    img_seg.dtype = 'bool'
    mask = np.ma.array(img_seg, mask=~img_seg)


#%% Main
# Main part

def Main():

    # Set path
    DataPathGrey = '/home/stefan/PycharmProjects/TBS/01_Data/SingleVertebrae/'
    DataPathSeg = '/home/stefan/PycharmProjects/TBS/01_Data/SegmentedImages/'
    Json = '/mnt/c/Users/mathi/Downloads/shift_list_no_torque.json'

    # Load data
    with open(Json) as File:
        data = js.load(File)

    my_dpi = 600
    mm = 1 / 25.4
    for img_name_json in list(data):
        img_name = img_name_json.split('.ISQ')[0] + '.mhd'

    img_grey_sitk = sitk.ReadImage(DataPathGrey + 'L1.mhd')
    img_seg_sitk = sitk.ReadImage(DataPathSeg + 'L1_seg.mhd')
    img_grey_np = np.transpose(sitk.GetArrayFromImage(img_grey_sitk), [2, 1, 0])
    img_seg_np = np.transpose(sitk.GetArrayFromImage(img_seg_sitk), [2, 1, 0])
    proj = np.sum(img_grey_np, axis=2)
    
    cog = ndimage.measurements.center_of_mass(proj)

    coo_not_torque_pix = np.array(data[img_name_json]) / np.array(header_grey['resolution'])
    proj_cog = cross(proj, cog)
    proj_cog_no_torque = big_cross(proj_cog, cog + coo_not_torque_pix[:2])
    proj_cog_no_torque_seg = proj_cog_no_torque + img_seg_np[:, :, -2] * 100000

    layers = 50
    slice_seg = img_seg_np[:, :, -2]
    slice_seg_aug = np.zeros((slice_seg.shape[0] + 2 * layers, slice_seg.shape[1] + 2 * layers))
    slice_seg_aug[layers:int(slice_seg.shape[0] + layers), layers:int(slice_seg.shape[1] + layers)] = slice_seg

    slice_grey = proj_cog_no_torque
    slice_grey_aug = np.zeros((slice_grey.shape[0] + 2 * layers, slice_grey.shape[1] + 2 * layers))
    slice_grey_aug[layers:int(slice_grey.shape[0] + layers), layers:int(slice_grey.shape[1] + layers)] = slice_grey
    slice_grey_aug_cog = cross(slice_grey_aug, cog + np.array([layers] * 2).astype(float))
    slice_grey_aug_cog_no_torque = big_cross(slice_grey_aug, cog + coo_not_torque_pix[:2] + layers,
                                            np.max(slice_grey_aug))
    
    # try out rotation to match actual orientation
    slice_grey_aug_cog_no_torque_rotated = np.rot90(slice_grey_aug_cog_no_torque, k=2, axes=(0, 1))
    # mask_rot = np.rot90(img_seg_np[:, :, -2], k=2, axes=(0, 1))
    mask_rot = np.rot90(slice_seg_aug, k=2, axes=(0, 1))
    
    # check if inverse works, otherwise do not invert
    mask_rot_int = mask_rot.astype(int)
    mask_rot_int_inv = np.ma.array(mask_rot_int, mask=mask_rot_int * (-1) + 1)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(slice_grey_aug.shape * np.array(header_grey['resolution'][:2]) / 25.4)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(slice_grey_aug_cog_no_torque_rotated, cmap='Greys_r',
            extent=[0, slice_grey_aug.shape[0] * header_grey['resolution'][0],
                    slice_grey_aug.shape[1] * header_grey['resolution'][0], 0], aspect='auto')
    ax.imshow(mask_rot_int_inv, cmap=plt.cm.flag, interpolation='none', vmin=0, vmax=np.max(slice_grey_aug),
            alpha=0.6, extent=[0, slice_grey_aug.shape[0] * header_grey['resolution'][0],
                                slice_grey_aug.shape[1] * header_grey['resolution'][0], 0], aspect='auto')
    plt.gca().add_artist(ScaleBar(1, 'mm'))
    plt.tight_layout()
    plt.savefig('../01_Data/6_Neck_paper_template/' + img_name.replace('mhd', 'pdf'),
                dpi=1 / (np.array(header_grey['resolution'][0]) / 25.4))
    print('\n')

            # slice_grey_aug.shape = (352, 344) in pixel
            # slice_grey_aug.shape * np.array(header_grey['resolution'][:2]) = array([46.18166667, 45.13208333]) in mm
            # inch = array([1.81817585, 1.77685367])
            # dpi = 193.6



#%% If main
if __name__ == '__main__':

    # Initiate the parser with a description
    Parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Add long and short argument
    ScriptVersion = Parser.prog + ' version ' + Version
    Parser.add_argument('-v', '--Version', help='Show script version', action='version', version=ScriptVersion)

    # Read arguments from the command line
    Arguments = Parser.parse_args()

    Main()
