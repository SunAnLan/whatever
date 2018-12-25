""" Implement of helpers to visualize the CT scan and slice

Additional libraries:
    matplotlib

Summary of available functions:
    vis_slice: 2D visualize for a slice
    vis_slices: 2D visualize for several slices
    view_scan: interactive 2D view for a volume
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import bbox as B

FIG_SIZE = (9, 9)
SCROLL_STEP = 1
KEY_STEP = 4


def vis_slice(slice, bbox_list=(), title=None, path=None, color=False, figsize=FIG_SIZE):
    """
    Visualize a slice
    :param slice: slice to display
    :param bbox_list: a list of bounding box to display
    :param title: title on the figure
    :param path: path to save the fig. If not None, save figure without display
    :param color: True for rgb, False for gray
    :param figsize: figure size
    :return: None
    """
    # open a new figure
    fig = plt.figure(figsize=figsize)

    # black or colorful
    if color:
        plt.imshow(slice)
    else:
        plt.imshow(slice, cmap=plt.cm.bone)

    # render bounding box
    ax = plt.gca()
    for bbox in bbox_list:
        rect = Rectangle((bbox[1]-1, bbox[0]-1), bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1,
                         fill=False, color='red', linewidth=1)
        ax.add_patch(rect)

    if title is not None:
        plt.title(title)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

    plt.close(fig)


def vis_slices(slices, ncols=3, nrows=None, title=None, path=None, color=False, figsize=(12,6)):
    """
    Visualize slices
    :param slices: tuple or list of slices, as long as it is iterable
    :param ncols: number of columns to display
    :param nrows: number of rows to display
    :param title:
    :param path:
    :param color:
    :param figsize:
    :return:
    """
    if nrows is None:
        nrows = np.ceil(len(slices)/float(ncols)).astype(np.int16)
    fig, plots = plt.subplots(nrows, ncols, figsize=figsize)

    if len(plots.shape) < 2:
        plots = np.expand_dims(plots, 0)

    for ind, image_slice in enumerate(slices):
        plots[int(ind / ncols), int(ind % ncols)].axis('off')
        if color:
            plots[int(ind / ncols), int(ind % ncols)].imshow(image_slice)
        else:
            plots[int(ind / ncols), int(ind % ncols)].imshow(image_slice, cmap=plt.cm.bone)

    if title is not None:
        plt.title(title)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

    plt.close(fig)

def sagittal(bbox, shape):
    sagibbox = [bbox[0], bbox[2]*shape[0]/shape[2], bbox[1], bbox[3], bbox[5]*shape[0]/shape[2], bbox[4]]
    return sagibbox


def view_scan(vol, bbox_list=(), attr_list=None, color_list=None, start_slice=0, color=False, figsize=FIG_SIZE, patientName = None):
    """
    Interactive view a volume
    Keyboard Mapping:
    w-next slice; z-previous slice;
    d-next bbox; a-previous bbox;
    mouse scroll- 5 slices to forward or backward
    :param vol: 3D numpy array
    :param bbox_list: a list of bounding box
    :param attr_list: optional. If not None, it must be of same length of bbox_list
    :param start_slice: first slice to display
    :param color:
    :return:
    """
    global cur_slice, bbox_flag, cur_bbox, windows, lowlevel, highlevel, sagittal_Cur_slice, bs, shape, patientID

    bbox_flag = True  # whether to display bounding boxes
    cur_bbox = None  # current bounding box index
    cur_slice = start_slice  # current slice to display
    windows = "f"
    bs = '.'
    shape = vol.shape
    patientID = patientName
    sagittal_Cur_slice = start_slice

    fig = plt.figure(figsize=figsize)

    # It seems plt would adjust the range of color by the first displayed image.
    # Thus, for label, we might display a nonzero slice first.
    if np.max(vol[:, :, cur_slice]) == np.min(vol[:, :, cur_slice]):
        slice = np.random.uniform(np.min(vol), np.max(vol), size=(vol.shape[0], vol.shape[1]))
    else:
        slice = vol[:, :, cur_slice]
    if color:

        im = plt.imshow(slice)
    else:
        lowlevel = np.min(slice)
        highlevel = np.max(slice)
        slice = (slice + 1150) * (slice + 1150 > 0)
        slice = (slice >= 1500) * 1500 + slice * (slice < 1500)
        slice = slice / 1500 * (highlevel - lowlevel) + lowlevel
        im = plt.imshow(slice, cmap=plt.cm.gray)
    plt.axis('off')
    plt.axis('equal')
    # height, width = slice.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    ax = plt.gca()
    ax.set_title('patientID=%s, z=%d' %(patientID, cur_slice))

    artists = list()
    bbox_list = [B.regularize(bbox) for bbox in bbox_list]
    vbbox_list = [B.regularize(sagittal(bbox, vol.shape)) for bbox in bbox_list]

    def sagittalShow(next_slice):
        global sagittal_Cur_slice, lowlevel, highlevel, bs,bbox_flag, cur_bbox, windows

        next_slice = int(next_slice)
        sagittal_Cur_slice = max(0, min(vol.shape[1] - 1, next_slice))
        imSlice = vol[:,sagittal_Cur_slice,:]
        if windows == "f":

            # imSlice = (imSlice>=-1150)*imSlice + (imSlice<-1150)*(-1150)
            # imSlice = (imSlice<=350)*imSlice + (imSlice>350)* 350
            # imSlice = (imSlice - (-1150))/1500*(highlevel - lowlevel) + lowlevel
            imSlice = (imSlice + 1150)*(imSlice + 1150> 0)
            imSlice = (imSlice >= 1500) * 1500 + imSlice*(imSlice < 1500)
            imSlice = imSlice/ 1500 * (highlevel - lowlevel) + lowlevel
        elif windows == 'z':
            imSlice = (imSlice + 160)*(imSlice + 160> 0)
            imSlice = (imSlice >= 380) * 380 + imSlice*(imSlice < 380)
            imSlice = imSlice/ 380 * (highlevel - lowlevel) + lowlevel
        elif windows == "o":
            pass

        im.set_array(imSlice)
        ax.set_title('patientID=%s, y=%d' % (patientID, sagittal_Cur_slice))
        for artist in artists:
            artist.remove()
        del artists[:]

        if bbox_flag:
            for bbox_id, bbox in enumerate(vbbox_list):
                bbox = (np.round(bbox)).astype(np.int16)
                if not color_list:
                    color_now = 'yellow'
                else:
                    color_now = color_list[bbox_id]
                if bbox[2] <= sagittal_Cur_slice < bbox[5]:
                    rect = Rectangle((bbox[1] - 1, bbox[0] - 1), bbox[4] - bbox[1] + 1, bbox[3] - bbox[0] + 1,
                                     fill=False, color=color_now, linewidth=1)
                    ax.add_patch(rect)
                    artists.append(rect)

                    anno = '%d:' % bbox_id
                    if attr_list is not None:
                        anno = anno + str(attr_list[bbox_id])
                    artists.append(ax.annotate(anno, xy=(bbox[4] + 5, bbox[3] + 5), color=color_now))

        plt.draw()


    def show(next_slice):
        """
        helpers to visualize slice
        :param next_slice: next slice to display, which will be assigned to cur slice
        :return: None
        """
        global cur_slice, bbox_flag, cur_bbox, windows, lowlevel, highlevel, bs

        next_slice = int(next_slice)
        cur_slice = max(0, min(vol.shape[2] - 1, next_slice))
        # print 'show: cur_slice=', cur_slice
        # print 'show: bbox_flag=', bbox_flag
        # print 'show: cur_bbox=', cur_bbox
        imSlice = vol[:, :, cur_slice]
        if windows == "f":

            # imSlice = (imSlice>=-1150)*imSlice + (imSlice<-1150)*(-1150)
            # imSlice = (imSlice<=350)*imSlice + (imSlice>350)* 350
            # imSlice = (imSlice - (-1150))/1500*(highlevel - lowlevel) + lowlevel
            imSlice = (imSlice + 1150)*(imSlice + 1150> 0)
            imSlice = (imSlice >= 1500) * 1500 + imSlice*(imSlice < 1500)
            imSlice = imSlice/ 1500 * (highlevel - lowlevel) + lowlevel
        elif windows == 'z':
            imSlice = (imSlice + 160)*(imSlice + 160> 0)
            imSlice = (imSlice >= 380) * 380 + imSlice*(imSlice < 380)
            imSlice = imSlice/ 380 * (highlevel - lowlevel) + lowlevel
        elif windows == "o":
            pass
        im.set_array(imSlice)
        ax.set_title('patientID=%s, z=%d' %(patientID, cur_slice))

        # clear bbox
        for artist in artists:
            artist.remove()
        del artists[:]

        if bbox_flag:
            for bbox_id, bbox in enumerate(bbox_list):
                bbox = (np.round(bbox)).astype(np.int16)
                if not color_list:
                    color_now = 'yellow'
                else:
                    color_now = color_list[bbox_id]
                if bbox[2] <= cur_slice < bbox[5]:
                    rect = Rectangle((bbox[1] - 1, bbox[0] - 1), bbox[4] - bbox[1] + 1, bbox[3] - bbox[0] + 1,
                                     fill=False, color=color_now, linewidth=1)
                    ax.add_patch(rect)
                    artists.append(rect)

                    anno = '%d:' % bbox_id
                    if attr_list is not None:
                        anno = anno + str(attr_list[bbox_id])
                    artists.append(ax.annotate(anno, xy=(bbox[4] + 5, bbox[3] + 5), color=color_now))

        plt.draw()  # python2 plt.draw() works while python3 fig.show() works

    def on_scroll(event):
        print(SCROLL_STEP)
        if bs == ',':
            sagittalShow(sagittal_Cur_slice + event.step * SCROLL_STEP)
        elif bs == '.':
            show(cur_slice + event.step * SCROLL_STEP)

    def button_release(event):
        global bs, sagittal_Cur_slice,  cur_slice, shape
        if bs == '.':
            (posY, posX)= plt.ginput()[0]
            sagittal_Cur_slice = posY
            # sagittalShow(sagittal_Cur_slice)
        if bs == ',':
            (posX, posZ) = plt.ginput()[0]
            cur_slice = posX*shape[2]/shape[0]
            # show(cur_slice)
        print(posX)



    def on_press(event):
        print(event.key)
        global cur_slice, bbox_flag, cur_bbox, windows, bs, sagittal_Cur_slice, shape

        if event.key == "x":
            print(KEY_STEP)
            show(cur_slice - KEY_STEP)
        elif event.key == "w":
            show(cur_slice + KEY_STEP)
        elif event.key == "a":
            if cur_bbox is not None:
                if 1 <= cur_bbox < len(bbox_list):
                    cur_bbox -= 1
                show(int((bbox_list[cur_bbox][2] + bbox_list[cur_bbox][5]) / 2))
        elif event.key == "d":
            if cur_bbox is None and len(bbox_list):
                cur_bbox = 0
            if 0 <= cur_bbox < len(bbox_list) - 1:
                cur_bbox += 1
            show(int((bbox_list[cur_bbox][2] + bbox_list[cur_bbox][5]) / 2))
        elif event.key == "b":
            bbox_flag = not bbox_flag
            show(cur_slice)
        elif event.key == "f":
            windows = 'f'
            if bs == '.':
                show(cur_slice)
            elif bs ==',':
                sagittalShow(sagittal_Cur_slice)
        elif event.key == "z":
            windows = 'z'
            if bs == '.':
                show(cur_slice)
            elif bs ==',':
                sagittalShow(sagittal_Cur_slice)
        elif event.key == "o":
            windows = 'o'
            if bs == '.':
                show(cur_slice)
            elif bs ==',':
                sagittalShow(sagittal_Cur_slice)
        elif event.key == ",":
            if bs != ',':
                (posY, posX)= plt.ginput()[0]
                sagittal_Cur_slice = posY
            bs = ','
            sagittalShow(sagittal_Cur_slice)

        elif event.key == ".":
            if bs != '.':
                (posX, posZ) = plt.ginput()[0]
                cur_slice = posX*shape[2]/shape[0]
            bs = '.'
            show(cur_slice)





    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_press)
    # fig.canvas.mpl_connect('button_release_event', button_release)
    plt.show()
    plt.close(fig)


# def vis_scan(vol, ncols=4, title=None, path=None, color=False):
#     num_slice = vol.shape[2]
#     nrows = int((num_slice+ncols-1)/ncols)
#     fig, plots = plt.subplots(nrows, ncols, figsize=FIG_SIZE)
#
#     for z in range(num_slice):
#         plots[int(z / ncols), int(z % ncols)].axis('off')
#         if color:
#             plots[int(z / ncols), int(z % ncols)].imshow(vol[:, :, z])
#         else:
#             plots[int(z / ncols), int(z % ncols)].imshow(vol[:, :, z], cmap=plt.cm.bone)
#
#     if title is not None:
#         plt.title(title)
#
#     if path is None:
#         plt.show()
#     else:
#         plt.savefig(path)
#
#     plt.close(fig)
