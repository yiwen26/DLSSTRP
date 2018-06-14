import numpy as np
from scipy import fftpack
from scipy import ndimage
from sklearn.preprocessing import OneHotEncoder
import cv2


"""FFT for idnetifying defects"""


def FFTmask(imgsrc, maskratio=10):
    """Takes a square real space image and filter out a disk with radius equal to:
    1/maskratio * image size.
    Retruns FFT transform of the image and the filtered FFT transform
    """
    # Take the fourier transform of the image.
    F1 = fftpack.fft2((imgsrc))
    # Now shift so that low spatial frequencies are in the center.
    F2 = (fftpack.fftshift((F1)))
    # copy the array and zero out the center
    F3 = F2.copy()
    l = int(imgsrc.shape[0] / maskratio)
    m = int(imgsrc.shape[0] / 2)
    y, x = np.ogrid[1: 2 * l + 1, 1:2 * l + 1]
    mask = (x - l) * (x - l) + (y - l) * (y - l) <= l * l
    F3[m - l:m + l, m - l:m + l] = F3[m - l:m + l, m - l:m + l] * (1 - mask)

    return F2, F3


def FFTsub(imgsrc, F3):
    """Takes real space image and filtred FFT.
    Reconstructs real space image and subtracts it from the original.
    Returns normalized image.
    """
    reconstruction = np.real(fftpack.ifft2(fftpack.ifftshift(F3)))
    diff = np.abs(imgsrc - reconstruction)

    # normalization
    diff = diff - np.amin(diff)
    diff = diff / np.amax(diff)

    return diff


"""Image thresholding"""


def threshImg(diff, threshL, threshH):
    """Takes in difference image, low and high thresold values, and outputs a map of all defects.
    """

    assert isinstance(threshL, float), ('Wrong data type',
                                        'threshL must be a float')
    assert isinstance(threshH, float), ('Wrong data type',
                                        'threshH must be a float')

    threshIL = diff < threshL
    threshIH = diff > threshH
    threshI = threshIL + threshIH

    return threshI


"""Generate Training Data"""

# Generate x, y positions for sliding windows


def GenerateXYPos(window_size, window_step, image_width):
    """Takes the window size, step, and total image width
    and generates all xy pairs for sliding window"""
    xpos_vec = np.arange(0, image_width - window_size, window_step)
    ypos_vec = np.arange(0, image_width - window_size, window_step)

    num_steps = len(xpos_vec)

    xpos_mat = np.tile(xpos_vec, num_steps)
    ypos_mat = np.repeat(ypos_vec, num_steps)
    pos_mat = np.column_stack((xpos_mat, ypos_mat))

    return pos_mat


def MakeWindow(imgsrc, xpos, ypos, window_size):
    """returns window of given size taken at position on image"""
    imgsrc = imgsrc[xpos:xpos + window_size, ypos:ypos + window_size]
    return imgsrc


def imgen(raw_image, pos_mat, window_size):
    """Returns all windows from image for given positions array"""

    immat = np.zeros(shape=(len(pos_mat), window_size, window_size))

    for i in np.arange(0, len(pos_mat)):
        img_window = MakeWindow(
            raw_image, pos_mat[i, 0], pos_mat[i, 1], window_size)
        immat[i, :, :, ] = img_window

    return immat


def image_preprocessing(image_data, norm=0):
    """Reshapes data and optionally normalizes it"""
    image_data = image_data.reshape(
        image_data.shape[0],
        image_data.shape[1],
        image_data.shape[2],
        1)
    image_data = image_data.astype('float32')
    if norm != 0:
        image_data = (image_data - np.amin(image_data)) / \
            (np.amax(image_data) - np.amin(image_data))
    return image_data


def label_preprocessing(image_data, nb_classes):
    """Returns labels / ground truth for images"""

    label4D = np.empty(
        (0,
         image_data.shape[1],
         image_data.shape[2],
         nb_classes))
    for idx in range(image_data.shape[0]):
        img = image_data[idx, :, :]
        n, m = img.shape
        img = np.array(OneHotEncoder(n_values=nb_classes).fit_transform(
            img.reshape(-1, 1)).todense())
        img = img.reshape(n, m, nb_classes)
        label4D = np.append(label4D, [img], axis=0)
    return label4D


def label_preprocessing2(image_data):
    """we can simplify this in case of only two classes"""

    label1 = image_data.reshape(
        image_data.shape[0],
        image_data.shape[1],
        image_data.shape[2],
        1)
    label2 = -label1 + 1
    label4D = np.concatenate((label2, label1), axis=3)
    return label4D


def splitImage(img, target_size):
    """Splits image into patches with given size"""

    xs = img.shape[0]
    ys = img.shape[1]
    nxp = int(xs / target_size[0])
    nyp = int(ys / target_size[1])

    impatchmat = np.zeros(
        shape=(int(nxp * nyp), target_size[0], target_size[1], 1))

    count = 0
    for i in range(nxp):
        for i2 in range(nyp):
            xstart = target_size[0] * i
            ystart = target_size[1] * i2
            xend = target_size[0] * (i + 1)
            yend = target_size[1] * (i2 + 1)

            impatchmat[count, :, :, 0] = img[xstart:xend, ystart:yend]
            count = count + 1

    return impatchmat


def predictDefects(img, model, target_size, nb_classes=2):
    """Uses given DL model to generate prediciton maps on the image"""

    xs = img.shape[0]
    ys = img.shape[1]
    nxp = int(xs / target_size[0])
    nyp = int(xs / target_size[1])
    classpred = np.zeros(shape=(nb_classes, xs, ys))

    impatchmat = splitImage(img, target_size)
    res = model.predict(impatchmat)

    count = 0
    for i in range(nxp):
        for i2 in range(nyp):
            xstart = target_size[0] * i
            ystart = target_size[1] * i2
            xend = target_size[0] * (i + 1)
            yend = target_size[1] * (i2 + 1)

            for i3 in range(nb_classes):
                classpred[i3, xstart:xend, ystart:yend] = res[count, :, :, i3]

            count = count + 1

    return classpred, res


"""Extract Defects"""


def extractDefects(img, classpred, softmax_threhold, bbox):

    defim = np.ndarray(shape=(0, bbox * 2, bbox * 2))
    defcoord = np.ndarray(shape=(0, 2))
    defcount = 0

    _, thresh = cv2.threshold(
        classpred[1], softmax_threhold, 1, cv2.THRESH_BINARY)

    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    labeled, nr_objects = ndimage.label(thresh, structure=s)
    loc = ndimage.find_objects(labeled)
    cc = ndimage.measurements.center_of_mass(
        labeled, labeled, range(nr_objects + 1))
    sizes = ndimage.sum(thresh, labeled, range(1, nr_objects + 1))

    # filter found points
    cc2 = cc[1:]
    t = zip(cc2, sizes)
    ccc = [k[0] for k in t if (k[1] < 700 and k[1] > 5)]

    max_coord = ccc

    for point in max_coord:

        startx = int(round(point[0] - bbox))
        endx = startx + bbox * 2
        starty = int(round(point[1] - bbox))
        endy = starty + bbox * 2

        if startx > 0 and startx < img.shape[0] - bbox * 2:
            if starty > 0 and starty < img.shape[1] - bbox * 2:

                defim.resize(defim.shape[0] + 1, bbox * 2, bbox * 2)
                defim[defcount] = img[startx:endx, starty:endy]
                defcoord.resize(defcoord.shape[0] + 1, 2)
                defcoord[defcount] = point[0:2]

                defcount = defcount + 1

    return thresh, defim, defcoord


def defectcropped(bbox, img, defcoord):
    """
    imgdata : the raw image file converted from dm3 video file
    """
    defect_list = []
    for point in defcoord:
        startx = int(round(point[0] - bbox))
        endx = startx + bbox * 2
        starty = int(round(point[1] - bbox))
        endy = starty + bbox * 2
        if startx < 0 or startx > 512:
            continue
        if endx < 0 or endx > 512:
            continue
        if starty < 0 or starty > 512:
            continue
        if endy < 0 or endy > 512:
            continue
        else:
            cropped = img[startx:endx, starty:endy]
            defect_list.append(cropped)
    return defect_list
