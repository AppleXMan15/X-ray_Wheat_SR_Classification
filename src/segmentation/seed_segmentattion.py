import cv2
import numpy as np
import os

def process_and_save_grains(image_path, output_folder="grains", grains_on_image=100, cur_image=0):
    # Создаем папку для сохранения зерен, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Шаг 1: Загрузка изображения
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # # Шаг 1.1: Убираем детали рентген-аппарата
    # center = (435, 1210)  # центр прямоугольника
    # width = 30  # ширина
    # height = 525  # длина
    # angle = -0.8  # угол (в градусах)
    # rect = ((center[0], center[1]), (width, height), angle)
    #
    # box = cv2.boxPoints(rect)  # получаем 4 точки
    # box = np.int32(box)
    # color = (37, 37, 37)
    # thickness = -1
    # cv2.drawContours(original_image, [box], 0, color, thickness)
    #
    # # Шаг 1.2: Кроп
    # x1, y1 = 420, 140  # верхний левый угол
    # x2, y2 = 1470, 1490  # нижний правый угол
    # original_image = original_image[y1:y2, x1:x2]

    # Шаг 2: Увеличение контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(original_image)

    # Шаг 3: Размытие для сглаживания контуров
    blurred_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)

    # Шаг 4: Бинаризация с использованием метода OTSU
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Шаг 5: Морфологические операции для улучшения формы объектов
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)

    # Шаг 6: Удаление мелких объектов (шумов) по площади
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    min_size = 100  # Минимальный размер объекта
    binary_image_cleaned = np.zeros(binary_image.shape, dtype=np.uint8)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            binary_image_cleaned[labels == i] = 255

    # Шаг 7: Дистанционное преобразование
    distance_transform = cv2.distanceTransform(binary_image_cleaned, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(distance_transform, 0.7 * distance_transform.max(), 255, 0)

    # Шаг 8: Создание маркеров для Watershed
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(binary_image_cleaned, sure_fg)
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Шаг 9: Визуализация результатов алгоритма Watershed
    output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # Шаг 10: Фильтрация контуров по площади
    contours, _ = cv2.findContours(binary_image_cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 1000  # Минимальная площадь контура
    max_contour_area = 9000  # Максимальная площадь контура
    filtered_contours = [cnt for cnt in contours
                         if (cv2.contourArea(cnt) > min_contour_area) and
                         (cv2.contourArea(cnt) < max_contour_area) and
                         (max(cv2.minAreaRect(cnt)[1])/min(cv2.minAreaRect(cnt)[1])<4)]

    # Шаг 11: Сохранение каждого зерна как отдельного изображения и нумерация
    for i, contour in enumerate(filtered_contours):
        # Получаем ограничивающий прямоугольник для каждого контура
        x, y, w, h = cv2.boundingRect(contour)
        # Отступ для полного захвата зерна
        bound = 5
        # Рисуем прямоугольник вокруг зерна
        cv2.rectangle(output_image, (x-bound, y-bound), (x + w + bound, y + h + bound), (0, 255, 0), 2)
        # Вырезаем область зерна из оригинального изображения
        grain = original_image[max(0,y-bound):y+h+bound, max(0,x-bound):x+w+bound]
        # Сохраняем зерно в папку с номером
        grain_path = os.path.join(output_folder, f"grain_{grains_on_image * cur_image + i + 1}.png")
        # Проверка
        # print(type(grain))
        # print(grain.size)
        # if grain is not None:
        #     print(grain.shape)
        cv2.imwrite(grain_path, grain)

        # Нумерация на итоговом изображении
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.putText(output_image, str(i + 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return output_image

for i in range(1, 121):
    image_path = f"{str(i).zfill(3)}.jpg"
    result_image = process_and_save_grains(image_path, grains_on_image=100, cur_image=i-1)
    cv2.imwrite("result.png", result_image)
    cv2.namedWindow(f'Custom Window {i}', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Custom Window {i}', 700, 800)
    cv2.imshow(f"Custom Window {i}", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # Пример использования функции
# image_path = "seeds_cut.jpg"
# result_image = process_and_save_grains(image_path)

# # Сохранение и отображение итогового изображения
# cv2.imwrite("result.png", result_image)
# cv2.namedWindow('Custom Window', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Custom Window', 700, 800)
# cv2.imshow("Custom Window", result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
