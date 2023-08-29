import pytesseract
import numpy as np
import cv2 as cv2

# your path may be different
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


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
img = cv2.imread("test02-02.jpg")

# change BGR to RGB
# rgb = img
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# nie polecany przez dokumentacja esposób wyboru jezyka
# text = pytesseract.image_to_string(rgb, lang='por')

# rekomendowany przez dokumentacje sposób wyboru języka
config_tesseract = '--tessdata-dir tessdata'
text = pytesseract.image_to_string(rgb, lang='por', config=config_tesseract)

print(text)

resize = ResizeWithAspectRatio(rgb, width=600)  # Resize by width OR
cv2.imshow('resize', resize)

cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image

print("bye")
