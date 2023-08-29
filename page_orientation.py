import pytesseract
import numpy as np
import cv2 as cv2
from PIL import Image
import matplotlib.pyplot as plt
from pytesseract import Output

# your path may be different
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Page segmentation modes:
#   0    Orientation and script detection (OSD) only.
#   1    Automatic page segmentation with OSD.
#   2    Automatic page segmentation, but no OSD, or OCR.
#   3    Fully automatic page segmentation, but no OSD. (Default)
#   4    Assume a single column of text of variable sizes.
#   5    Assume a single uniform block of vertically aligned text.
#   6    Assume a single uniform block of text.
#   7    Treat the image as a single text line.
#   8    Treat the image as a single word.
#   9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
#        bypassing hacks that are Tesseract-specific.
##############################################################################
# DEF
############


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def bounding_box(result, img, i, color=(255, 100, 0)):
    x = result['left'][i]
    y = result['top'][i]
    w = result['width'][i]
    h = result['height'][i]

    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    return x, y, img


def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)
#######################################################################
# MAIN
#######################################################################


print("hello")

# reading the image
# img = cv2.imread("page-book.jpg")
img = cv2.imread("test01.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# resize = ResizeWithAspectRatio(rgb, width=600)  # Resize by width OR
# cv2.imshow('resize', resize)

config_tesseract = '--tessdata-dir tessdata'
result = pytesseract.image_to_data(
    rgb, config=config_tesseract, lang='eng', output_type=Output.DICT)

print(result)

min_confidence = 40
img_copy = rgb.copy()
for i in range(0, len(result['text'])):
    confidence = int(result['conf'][i])
    if confidence > min_confidence:
        # print(confidence)
        x, y, img = bounding_box(result, img_copy, i)
        # print(x, y)
        text = result['text'][i]
        cv2.putText(img_copy, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255))
resize = ResizeWithAspectRatio(img_copy, width=600)  # Resize by width OR
cv2.imshow("test", resize)
showInMovedWindow('test', resize, 0, 200)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image

print("bye")

# block_num = Current block number. When Tesseract performs the detections,
#  it divides the image into several regions, which can vary according to the PSM parameters and also other criteria of the algorithm. Each block is a region

# conf = prediction confidence (from 0 to 100. -1 means no text was recognized)

# height = height of detected block of text (bounding box)

# left = x coordinate where the bounding box starts

# level = the level corresponds to the category of the detected block. There are 5 possible values:

# page
# block
# paragraph
# line
# word
# Therefore, if 5 is returned, it means that the detected block is text, if it was 4, it means that a line was detected

# line_num = line number (starts from 0)

# page_num = the index of the page where the item was detected

# text = the recognition result

# top = y-coordinate where the bounding box starts

# width = width of the current detected text block

# word_num = word number (index) within the current block
