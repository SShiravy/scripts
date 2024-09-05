import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from functools import wraps
from codetiming import Timer
from paddleocr import PaddleOCR


# -----------------------------------------------------------------------------
# execution timing
def execution_timing(msg):
    def timing(_func):
        @wraps(_func)
        def wrapper(*args, **kwargs):
            with Timer(text=msg + '->{:.1f}s'):
                value = _func(*args, **kwargs)
            return value

        return wrapper

    return timing


# -----------------------------------------------------------------------------


# open and crop image
def image_crop(img_path: str, corners_ratio: tuple):
    """
    corners_ratio in form (left,top,right,bottom)
    """
    img = Image.open(img_path)
    width, high = img.size
    left, top, right, bottom = (width * corners_ratio[0], high * corners_ratio[1],
                                width * corners_ratio[2], high * corners_ratio[3])

    cropped_img = img.crop((left, top, right, bottom))
    return img, cropped_img


# extract page's name
@execution_timing('extracting name')
def extract_text(img, lang: list = 'en') -> str:
    """
    extract text from image
    """
    img_np = np.array(img)
    ocr = PaddleOCR(lang=lang, show_log=False)
    result = ocr.ocr(img_np, cls=False)

    return result[0][0][-1][0]


@execution_timing('total elapsed time: ')
def main():
    if input_excel[-4:] == 'xlsx':
        try:
            all_names = pd.read_excel(input_excel)['names'].tolist()
        except:
            print(f"can't read{input_excel}")
            all_names = []
    else:
        all_names = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            img_path = os.path.join(input_dir, file_name)

            img, cropped_img = image_crop(img_path,
                                          (LEFT_ratio, TOP_ratio, RIGHT_ratio, BOTTOM_ratio))
            # img.show()
            # cropped_img.show()
            page_name = extract_text(cropped_img)
            # save image in output dir with new name
            if page_name and page_name not in all_names:
                img_out_path = os.path.join(output_dir, page_name) + file_name[file_name.find('.'):]
                all_names.append(page_name)
                print(page_name)
                img.save(img_out_path)
            else:
                print('nothing saved ---', page_name)
        else:
            print('wrong file format:', file_name)

    df = pd.DataFrame(all_names, columns=['names'])
    df.to_excel(f'names_output{len(all_names)}.xlsx', index=False)


if __name__ == '__main__':
    # create output dir if it is not existed
    LEFT_ratio = 0.11
    TOP_ratio = 0.04
    RIGHT_ratio = 0.89
    BOTTOM_ratio = 0.1
    # ----------------
    input_excel = input('enter excel name(ex: insta.xlsx):')  # 'names_output43.xlsx'
    input_dir = input('enter input images directory:')  # 'img'
    output_dir = input('enter output images directory:')  # 'renamed_img'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main()
    shutil.rmtree(input_dir) if input('enter y to clear input img dir:') == 'y' else None
    input('(enter to exit)')

# pyinstaller --collect-all pyclipper --collect-all skimage --collect-all paddleocr --collect-all imgaug --collect-all lmdb --collect-all requests --collect-all paddle OCR_script.py
