import cv2
import os
import pathlib
from pathlib import Path
import easyocr

os.environ['TESSDATA_PREFIX'] = "/home/wolf/mygitcode/Yong/deeplearning/doc_ocr/tesseract/tessdata"
print(os.environ['TESSDATA_PREFIX'])
parser = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory


def captch_ex(file_name):
    img = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
    cv2.imwrite("image_final.jpg", image_final)
    ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    cv2.imwrite("new_img.jpg", new_img)
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,
                                                         3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(new_img, kernel, iterations=9)  # dilate , more the iteration more the dilation

    cv2.imwrite("dilated.jpg", dilated)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # findContours returns 3 variables for getting contours

    index = 0
    for contour in contours:
        # get rectangle bounding contour
        [x, y, w, h] = cv2.boundingRect(contour)

        # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        basename = Path(file_name).stem
        crop_dir = Path("./dqd/").joinpath(basename)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_file_path = crop_dir.joinpath('crop_' + str(index) + '.jpg')

        cropped = img_final[y:y + h, x: x + w]

        crop_out_filepath = str(crop_file_path.resolve())
        cv2.imwrite(crop_out_filepath, cropped)

        result = parser.readtext(crop_out_filepath, text_threshold=0.3)
        print(crop_file_path.stem, result)

        index = index + 1

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk
    cv2.imwrite("result.jpg", img)
    # cv2.imshow('captcha_result', img)

    # cv2.waitKey()


file_name = './dqd/dqd_stat.jpg'
captch_ex(file_name)
