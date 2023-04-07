import cv2
import numpy as np
import os
import sys

# получаем путь к директории и имя файла с шаблоном Вальдо из аргументов командной строки
photos_dir = sys.argv[1]
template_file = sys.argv[2]

# загружаем шаблон Вальдо
template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)

# перебираем все файлы в директории
for filename in os.listdir(photos_dir):
    # проверяем тип файла
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    # загружаем изображение головоломки
    puzzle = cv2.imread(os.path.join(photos_dir, filename), cv2.IMREAD_GRAYSCALE)

    # ищем шаблон Вальдо в головоломке
    result = cv2.matchTemplate(puzzle, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    # вырезаем область с Вальдо из головоломки и помещаем ее на затемненный фон
    w, h = template.shape[::-1]
    mask = np.zeros_like(puzzle)
    mask[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w] = 255
    puzzle_with_waldo = np.zeros_like(puzzle)
    puzzle_with_waldo[mask == 255] = puzzle[mask == 255]
    puzzle_with_waldo[mask != 255] = 128

    # выводим название файла и коэффициент совпадения шаблона
    print(f'{filename}: {max_val}')

    # выводим изображение с выделенным шаблоном Вальдо, если коэффициент совпадения больше порогового значения
    if max_val >= 0.8:
        cv2.rectangle(puzzle, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 0), 2)
        cv2.imshow(filename, puzzle)
        cv2.waitKey(0)

cv2.destroyAllWindows()