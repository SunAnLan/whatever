import json
import os
import numpy as np
import pydicom as dicom
import SimpleITK as sitk
import re
import random
from viewer import vis_slice, view_scan

def SearchDir(dirname):
    """
    :param initial path of directory:
    :return the path saved the CT dicom files.:
    """
    dirlist = os.listdir(dirname)
    dirlist = [item for item in dirlist if os.path.isdir(os.path.join(dirname,item))]
    lengthOfFile = len(os.listdir(dirname)) - len(dirlist)
    allFile = all([os.path.isfile('/'.join([dirname, name])) for name in os.listdir(dirname)])
    if (lengthOfFile > 100) & allFile:
        return dirname
    elif allFile:
        return False
    else:
        for folderName in dirlist:
            filename = '/'.join([dirname, folderName])
            temp = SearchDir(filename)
            if type(temp) == str:
                return temp
            else:
                pass
def loadDicom(dir):
    """
    load the CT image sequences.
    :param dir: path of folder save the CT images.
    :return: the CT three dimenal voxel array.
    """
    DicomDir = SearchDir(dir)
    spacingList = []
    positionList =[]
    names = os.listdir(DicomDir)
    names = [ x for x in names if '.dcm' in x]
    for i, item in enumerate(names):
        fileName = os.path.join(DicomDir,item)
        imInfo= dicom.read_file(fileName, stop_before_pixels=True)
        positionList += [float(imInfo.ImagePositionPatient[2])]
    positionList = np.sort(np.array(positionList))
    try:

        for i in range(1, len(positionList)):
            temp = positionList[i] - positionList[0]
            temp = abs(temp) / abs((positionList[1] - positionList[0]))
            assert(abs(temp - round(temp))<1.0e-4)

    except:
        print(DicomDir)
        return False
       
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(DicomDir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    image_array = sitk.GetArrayFromImage(image) #zyx
    image_array = np.moveaxis(image_array, 0, -1) #yxz
    image_array = image_array.astype(np.float)
    origin = image.GetOrigin()  # xyz
    origin = (origin[1], origin[0], origin[2])  # yxz
    spacing = image.GetSpacing()  # xyz
    spacing = (spacing[1], spacing[0], spacing[2])  # yxz
    listDirection = list(image.GetDirection())
    assert len(listDirection) == 9
    assert all(listDirection[i] == 0.0 for i in range(len(listDirection)) if i not in [0, 4, 8])
    numpyDirection = np.array([listDirection[4], listDirection[0], listDirection[8]])
    # sumDirection = numpyDirection.sum()
    image_dict = dict(array=image_array, origin=origin, spacing=spacing, direction=numpyDirection)

    if not hasattr(imInfo, 'PatientAge') or imInfo.PatientAge is '':
        image_dict['age'] = random.randint(50, 70)
    else:
        image_dict['age'] = int(re.findall(r"[1-9]\d+", imInfo.PatientAge)[0])


    if not hasattr(imInfo, 'sex') or imInfo.PatientSex is '':
        image_dict['sex'] = random.randint(0, 1)
    elif imInfo.PatientSex is 'F':
        image_dict['sex'] = 0
    elif imInfo.PatientSex is 'M':
        image_dict['sex'] = 1


    # with open('some.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(image_array[0])
    print('age and sex ', image_dict['age'], image_dict['sex'])
    return image_dict

if __name__ == '__main__':
    folder = SearchDir('/Users/sunanlan/Downloads/viewer 2')
    im_dict = loadDicom(folder)
    # print(im_dict)
    with open('label_bbox_malignancy.json', 'r') as f:
        label = json.load(f)
    bboxlist = []
    attr_list = []
    color_list = []
    patientName = 'CT590466'
    for item in label[patientName]:
        if item != 'folder':
            nodule = label['CT590466'][item]
            # bboxlist.append([nodule['bbox'][4], nodule['bbox'][3], nodule['bbox'][1], nodule['bbox'][0], nodule['bbox'][2], nodule['bbox'][5]])
            bboxlist.append(nodule['bbox'])
            if (nodule['Docmalignancy']>0.5) == True:
                attr_list.append(1)
                color_list.append('red')
            else:
                attr_list.append(0)
                color_list.append('green')

        # vis_slice(im_dict['array'][:,:,int(nodule['position'][2])], )
    view_scan(im_dict['array'][:,:,::-1], bboxlist, attr_list, color_list, patientName=patientName)
    print('finished')
