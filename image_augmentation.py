import cv2
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np

import os

# co-relation between Opencv and Pillow Image Rectangle box
# (x1, y1) (left, top)
# (right, bottom) (x2, y2)

# (top,right,bottom,left)
# (32,64,0,0)

Folder_name = "C:/Users/ahmtb/capture_enroll_augment/augmented_images/"


# Extension = ".jpg"


# RESIZE
def resize_image(image, file_name, w, h):
    image = cv2.resize(image, (w, h))
    cv2.imwrite(Folder_name + "/Resize-" + str(w) + "*" + str(h) + file_name, image)


# crop
def crop_image(image, file_name, y1, y2, x1, x2):
    image = image[y1:y2, x1:x2]
    cv2.imwrite(Folder_name + "/Crop-" + str(x1) + str(x2) + "*" + str(y1) + str(y2) + file_name,
                image)


def padding_image(image, file_name, topBorder, bottomBorder, leftBorder, rightBorder, color_of_border=[0, 0, 0]):
    image = cv2.copyMakeBorder(image, topBorder, bottomBorder, leftBorder,
                               rightBorder, cv2.BORDER_CONSTANT, value=color_of_border)
    cv2.imwrite(
        Folder_name + "/padd-" + str(topBorder) + str(bottomBorder) + "*" + str(leftBorder) + str(
            rightBorder) + file_name, image)


def flip_image(image, file_name, dir):
    image = cv2.flip(image, dir)
    cv2.imwrite(Folder_name + "/flip-" + str(dir) + file_name, image)


def superpixel_image(image, file_name, segments):
    seg = segments

    def segment_colorfulness(image, mask):
        # split the image into its respective RGB components, then mask
        # each of the individual RGB channels, so we can compute
        # statistics only for the masked region
        (B, G, R) = cv2.split(image.astype("float"))
        R = np.ma.masked_array(R, mask=mask)
        G = np.ma.masked_array(B, mask=mask)
        B = np.ma.masked_array(B, mask=mask)

        # compute rg = R - G
        rg = np.absolute(R - G)

        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        # compute the mean and standard deviation of both `rg` and `yb`,
        # then combine them
        stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))

        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)

    orig = cv2.imread(image)
    vis = np.zeros(orig.shape[:2], dtype="float")

    # load the image and apply SLIC superpixel segmentation to it via
    # scikit-image
    image = io.imread(image)
    segments = slic(img_as_float(image), n_segments=segments,
                    slic_zero=True)
    for v in np.unique(segments):
        # construct a mask for the segment so we can compute image
        # statistics for *only* the masked region
        mask = np.ones(image.shape[:2])
        mask[segments == v] = 0

        # compute the superpixel colorfulness, then update the
        # visualization array
        C = segment_colorfulness(orig, mask)
        vis[segments == v] = C
    # scale the visualization image from an unrestricted floating point
    # to unsigned 8-bit integer array, so we can use it with OpenCV and
    # display it to our screen
    vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")

    # overlay the superpixel colorfulness visualization on the original
    # image
    alpha = 0.6
    overlay = np.dstack([vis] * 3)
    output = orig.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # cv2.imshow("Visualization", vis)
    cv2.imwrite(Folder_name + "/superpixels-" + str(seg) + file_name, output)


def invert_image(image, file_name, channel):
    # image=cv2.bitwise_not(image)
    image = (channel - image)
    cv2.imwrite(Folder_name + "/invert-" + str(channel) + file_name, image)


def add_light(image, file_name, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image = cv2.LUT(image, table)
    if gamma >= 1:
        cv2.imwrite(Folder_name + "/light-" + str(gamma) + file_name, image)
    else:
        cv2.imwrite(Folder_name + "/dark-" + str(gamma) + file_name, image)


def add_light_color(image, file_name, color, gamma=1.0):
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    image = cv2.LUT(image, table)
    if gamma >= 1:
        cv2.imwrite(Folder_name + "/light_color-" + str(gamma) + file_name, image)
    else:
        cv2.imwrite(Folder_name + "/dark_color" + str(gamma) + file_name, image)


def saturation_image(image, file_name, saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/saturation-" + str(saturation) + file_name, image)


def hue_image(image, file_name, saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/hue-" + str(saturation) + file_name, image)


def multiply_image(image, file_name, R, G, B):
    image = image * [R, G, B]
    cv2.imwrite(Folder_name + "/Multiply-" + str(R) + "*" + str(G) + "*" + str(B) + file_name, image)


def gausian_blur(image, file_name, blur):
    image = cv2.GaussianBlur(image, (5, 5), blur)
    cv2.imwrite(Folder_name + "/GausianBLur-" + str(blur) + file_name, image)


def averageing_blur(image, file_name, shift):
    image = cv2.blur(image, (shift, shift))
    cv2.imwrite(Folder_name + "/AverageingBLur-" + str(shift) + file_name, image)


def median_blur(image, file_name, shift):
    image = cv2.medianBlur(image, shift)
    cv2.imwrite(Folder_name + "/MedianBLur-" + str(shift) + file_name, image)


def bileteralBlur(image, file_name, d, color, space):
    image = cv2.bilateralFilter(image, d, color, space)
    cv2.imwrite(
        Folder_name + "/BileteralBlur-" + str(d) + "*" + str(color) + "*" + str(space) + file_name,
        image)


def erosion_image(image, file_name, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    cv2.imwrite(Folder_name + "/Erosion-" + "*" + str(shift) + file_name, image)


def dilation_image(image, file_name, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(Folder_name + "/Dilation-" + "*" + str(shift) + file_name, image)


def opening_image(image, file_name, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(Folder_name + "/Opening-" + "*" + str(shift) + file_name, image)


def closing_image(image, file_name, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(Folder_name + "/Closing-" + "*" + str(shift) + file_name, image)


def morphological_gradient_image(image, file_name, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite(Folder_name + "/Morphological_Gradient-" + "*" + str(shift) + file_name, image)


def top_hat_image(image, file_name, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite(Folder_name + "/Top_Hat-" + "*" + str(shift) + file_name, image)


def black_hat_image(image, file_name, shift):
    kernel = np.ones((shift, shift), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    cv2.imwrite(Folder_name + "/Black_Hat-" + "*" + str(shift) + file_name, image)


def sharpen_image(image, file_name):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    cv2.imwrite(Folder_name + "/Sharpen-" + file_name, image)


def emboss_image(image, file_name):
    kernel_emboss_1 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    image = cv2.filter2D(image, -1, kernel_emboss_1) + 128
    cv2.imwrite(Folder_name + "/Emboss-" + file_name, image)


def edge_image(image, file_name, ksize):
    image = cv2.Sobel(image, cv2.CV_16U, 1, 0, ksize=ksize)
    cv2.imwrite(Folder_name + "/Edge-" + str(ksize) + file_name, image)


def addeptive_gaussian_noise(image, file_name, ):
    h, s, v = cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    image = cv2.merge([h, s, v])
    cv2.imwrite(Folder_name + "/Addeptive_gaussian_noise-" + file_name, image)


def contrast_image(image, file_name, contrast):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:, :, 2] = [[max(pixel - contrast, 0) if pixel < 190 else min(pixel + contrast, 255) for pixel in row] for row
                      in image[:, :, 2]]
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(Folder_name + "/Contrast-" + str(contrast) + file_name, image)


def edge_detect_canny_image(image, file_name, th1, th2):
    image = cv2.Canny(image, th1, th2)
    cv2.imwrite(Folder_name + "/Edge Canny-" + str(th1) + "*" + str(th2) + file_name, image)


def grayscale_image(image, file_name):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(Folder_name + "/Grayscale-" + file_name, image)


def scale_image(image, file_name, fx, fy):
    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(Folder_name + "/Scale-" + str(fx) + str(fy) + file_name, image)


def translation_image(image, file_name, x, y):
    rows, cols, c = image.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Translation-" + str(x) + str(y) + file_name, image)


def rotate_image(image, file_name, deg):
    rows, cols, c = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Rotate-" + str(deg) + file_name, image)


def transformation_image(image, file_name):
    rows, cols, ch = image.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(1) + file_name, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [0, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(2) + file_name, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [30, 175]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(3) + file_name, image)

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 10], [200, 50], [70, 150]])
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(Folder_name + "/Transformations-" + str(4) + file_name, image)


def augment():
    scan = os.scandir("C:/Users/ahmtb/capture_enroll_augment/cut_outed_images")
    for s in scan:
        image_file = "C:/Users/ahmtb/capture_enroll_augment/cut_outed_images/" + s.name
        image = cv2.imread(image_file)

        resize_image(image, s.name, 450, 400)

        crop_image(image, s.name, 100, 400, 0, 350)  # (y1,y2,x1,x2)(bottom,top,left,right)
        crop_image(image, s.name, 100, 400, 100, 450)  # (y1,y2,x1,x2)(bottom,top,left,right)
        crop_image(image, s.name, 0, 300, 0, 350)  # (y1,y2,x1,x2)(bottom,top,left,right)
        crop_image(image, s.name, 0, 300, 100, 450)  # (y1,y2,x1,x2)(bottom,top,left,right)
        crop_image(image, s.name, 100, 300, 100, 350)  # (y1,y2,x1,x2)(bottom,top,left,right)

        padding_image(image, s.name, 100, 0, 0, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
        padding_image(image, s.name, 0, 100, 0, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
        padding_image(image, s.name, 0, 0, 100, 0)  # (y1,y2,x1,x2)(bottom,top,left,right)
        padding_image(image, s.name, 0, 0, 0, 100)  # (y1,y2,x1,x2)(bottom,top,left,right)
        padding_image(image, s.name, 100, 100, 100, 100)  # (y1,y2,x1,x2)(bottom,top,left,right)

        flip_image(image, s.name, 0)  # horizontal
        flip_image(image, s.name, 1)  # vertical
        flip_image(image, s.name, -1)  # both

        superpixel_image(image_file, s.name, 100)
        superpixel_image(image_file, s.name, 50)
        superpixel_image(image_file, s.name, 25)
        superpixel_image(image_file, s.name, 75)
        superpixel_image(image_file, s.name, 200)

        invert_image(image, s.name, 255)
        invert_image(image, s.name, 200)
        invert_image(image, s.name, 150)
        invert_image(image, s.name, 100)
        invert_image(image, s.name, 50)

        add_light(image, s.name, 1.5)
        add_light(image, s.name, 2.0)
        add_light(image, s.name, 2.5)
        add_light(image, s.name, 3.0)
        add_light(image, s.name, 4.0)
        add_light(image, s.name, 5.0)
        add_light(image, s.name, 0.7)
        add_light(image, s.name, 0.4)
        add_light(image, s.name, 0.3)
        add_light(image, s.name, 0.1)

        add_light_color(image, s.name, 255, 1.5)
        add_light_color(image, s.name, 200, 2.0)
        add_light_color(image, s.name, 150, 2.5)
        add_light_color(image, s.name, 100, 3.0)
        add_light_color(image, s.name, 50, 4.0)
        add_light_color(image, s.name, 255, 0.7)
        add_light_color(image, s.name, 150, 0.3)
        add_light_color(image, s.name, 100, 0.1)

        saturation_image(image, s.name, 50)
        saturation_image(image, s.name, 100)
        saturation_image(image, s.name, 150)
        saturation_image(image, s.name, 200)

        hue_image(image, s.name, 50)
        hue_image(image, s.name, 100)
        hue_image(image, s.name, 150)
        hue_image(image, s.name, 200)

        multiply_image(image, s.name, 0.5, 1, 1)
        multiply_image(image, s.name, 1, 0.5, 1)
        multiply_image(image, s.name, 1, 1, 0.5)
        multiply_image(image, s.name, 0.5, 0.5, 0.5)

        multiply_image(image, s.name, 0.25, 1, 1)
        multiply_image(image, s.name, 1, 0.25, 1)
        multiply_image(image, s.name, 1, 1, 0.25)
        multiply_image(image, s.name, 0.25, 0.25, 0.25)

        multiply_image(image, s.name, 1.25, 1, 1)
        multiply_image(image, s.name, 1, 1.25, 1)
        multiply_image(image, s.name, 1, 1, 1.25)
        multiply_image(image, s.name, 1.25, 1.25, 1.25)

        multiply_image(image, s.name, 1.5, 1, 1)
        multiply_image(image, s.name, 1, 1.5, 1)
        multiply_image(image, s.name, 1, 1, 1.5)
        multiply_image(image, s.name, 1.5, 1.5, 1.5)

        gausian_blur(image, s.name, 0.25)
        gausian_blur(image, s.name, 0.50)
        gausian_blur(image, s.name, 1)
        gausian_blur(image, s.name, 2)
        gausian_blur(image, s.name, 4)

        averageing_blur(image, s.name, 5)
        averageing_blur(image, s.name, 4)
        averageing_blur(image, s.name, 6)

        median_blur(image, s.name, 3)
        median_blur(image, s.name, 5)
        median_blur(image, s.name, 7)

        bileteralBlur(image, s.name, 9, 75, 75)
        bileteralBlur(image, s.name, 12, 100, 100)
        bileteralBlur(image, s.name, 25, 100, 100)
        bileteralBlur(image, s.name, 40, 75, 75)

        erosion_image(image, s.name, 1)
        erosion_image(image, s.name, 3)
        erosion_image(image, s.name, 6)

        dilation_image(image, s.name, 1)
        dilation_image(image, s.name, 3)
        dilation_image(image, s.name, 5)

        opening_image(image, s.name, 1)
        opening_image(image, s.name, 3)
        opening_image(image, s.name, 5)

        closing_image(image, s.name, 1)
        closing_image(image, s.name, 3)
        closing_image(image, s.name, 5)

        morphological_gradient_image(image, s.name, 5)
        morphological_gradient_image(image, s.name, 10)
        morphological_gradient_image(image, s.name, 15)

        top_hat_image(image, s.name, 200)
        top_hat_image(image, s.name, 300)
        top_hat_image(image, s.name, 500)

        black_hat_image(image, s.name, 200)
        black_hat_image(image, s.name, 300)
        black_hat_image(image, s.name, 500)

        sharpen_image(image, s.name)
        emboss_image(image, s.name)

        edge_image(image, s.name, 1)
        edge_image(image, s.name, 3)
        edge_image(image, s.name, 5)
        edge_image(image, s.name, 9)

        addeptive_gaussian_noise(image, s.name)

        contrast_image(image, s.name, 25)
        contrast_image(image, s.name, 50)
        contrast_image(image, s.name, 100)

        edge_detect_canny_image(image, s.name, 100, 200)
        edge_detect_canny_image(image, s.name, 200, 400)

        grayscale_image(image, s.name)

        scale_image(image, s.name, 0.3, 0.3)
        scale_image(image, s.name, 0.7, 0.7)
        scale_image(image, s.name, 2, 2)
        scale_image(image, s.name, 3, 3)

        translation_image(image, s.name, 150, 150)
        translation_image(image, s.name, -150, 150)
        translation_image(image, s.name, 150, -150)
        translation_image(image, s.name, -150, -150)

        rotate_image(image, s.name, 90)
        rotate_image(image, s.name, 180)
        rotate_image(image, s.name, 270)

        transformation_image(image, s.name)
