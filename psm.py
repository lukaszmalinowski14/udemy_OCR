import pytesseract
import numpy as np
import cv2 as cv2

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

#######################################################################
# MAIN
#######################################################################


print("hello")

# reading the image
# img = cv2.imread("page-book.jpg")
img = cv2.imread("exit.jpg")

# change BGR to RGB
# rgb = img
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# nie polecany przez dokumentacja esposób wyboru jezyka
# text = pytesseract.image_to_string(rgb, lang='por')

# rekomendowany przez dokumentacje sposób wyboru języka
config_tesseract = '--tessdata-dir tessdata --psm 8'
text = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)

print(text)

resize = ResizeWithAspectRatio(rgb, width=600)  # Resize by width OR
cv2.imshow('resize', resize)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image

print("bye")
