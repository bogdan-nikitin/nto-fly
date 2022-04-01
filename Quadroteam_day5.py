import rospy
from pyzbar import pyzbar
from sensor_msgs.msg import Image
import math
from cv_bridge import CvBridge
from clover import srv
from std_srvs.srv import Trigger
import cv2
import numpy as np
from sensor_msgs.msg import Range
from dataclasses import dataclass
from contextlib import suppress


# Константы

# Высота полёта над линией в метрах
LINE_FLY_HEIGHT = 0.75
# разрещение камеры дрона
SIZE = 240, 320  # HEIGHT, WIDTH
# минимальное расстояние по aruco_map между дефектами 
DEFECT_MIN_DIST = 0.8
# минимальная площадь дефекта в пикселях
MIN_DEFECT_AREA = 20
MIN_LINE_AREA = 400
DEFECT_CONTOUR_COLOR = (180, 105, 255)
OIL_CONTOUR_COLOR = (255, 0, 0)
CONTOUR_WIDTH = 3
QR_K = 0.005  # коэффициент для посадки по QR-коду
T = 2.384789007037547  # коэффициент для расчёта площади

# HSV пороги
DEFECT_HSV = (35, 200, 20), (60, 255, 50)
LINE_HSV = (30, 30, 160), (60, 200, 250)
OIL_HSV = (0, 80, 130), (20, 180, 180)

rospy.init_node('flight')

# Прокси
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)

bridge = CvBridge()

# Паблишеры
line_pub = rospy.Publisher('line', Image, queue_size=1)
defect_pub = rospy.Publisher('defect_detect', Image, queue_size=1)
oil_pub = rospy.Publisher('oil_detect', Image, queue_size=1)
red_pub = rospy.Publisher('red', Image, queue_size=1)

# Флаги
following_line = False
rotated = False
lost_qr = False
landing = False

# Данные с QR кода
qr_data = None
# Номер последней фотки
photo_num = 1
# Высота дрона над землёй, обновляется в range_callback
drone_height = 0
# найденные дефекты
defects = []


# Класс данных о пробое
@dataclass
class Defect:
    screen_dist: float  # расстояние до центра изображения (в пикселях)
    map_pos: (float, float)  # позиция на aruco карте
    oil_area: float


def get_contour_center(cnt):
    # центр контура на изображении
    moment = cv2.moments(cnt)
    x = int(moment['m10'] / moment['m00'])
    y = int(moment['m01'] / moment['m00'])
    return x, y


def get_distance(p1, p2):
    # дистанция между двумя точками в двумерной плоскости
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def update_defect_pos(screen_pos, map_pos, oil_area):
    # функция для вывода и сохранения координат разливов
    global defects
    screen_center = SIZE[0] / 2, SIZE[1] / 2

    # считаем расстояние от позиции разлива на изображении до центра
    # изображения
    screen_dist = get_distance(screen_center, screen_pos)
    # проверяем, не находили ли мы этот дефект ранее
    for defect in defects:
        # если разница в позиции меньше минимальной, то мы находили дефект
        # ранее
        if get_distance(map_pos, defect.map_pos) < DEFECT_MIN_DIST:
            # если расстояние до центра изображения меньше текущей, обновляем
            # позицию на более точную и обновляем площадь разлива
            if screen_dist < defect.screen_dist:
                defect.screen_dist = screen_dist
                defect.map_pos = map_pos
                defect.oil_area = oil_area
            return
    # если такого пробоя не было, добавляем его в список
    defects.append(Defect(screen_dist, map_pos, oil_area))
    # и по регламенту выводим на экран
    print(f'defect: {map_pos[0]} {map_pos[1]}')
    if oil_area > 0:
        print(f'oil area: {oil_area}')


def range_callback(msg):
    # обработка новых данных с дальномера
    global drone_height
    if msg.range != float('inf') and msg.range != float('nan'):
        drone_height = msg.range


def take_photo():
    # Снятие фото с камеры для дебага
    global photo_num
    img = bridge.imgmsg_to_cv2(
        rospy.wait_for_message(
            'main_camera/image_raw_throttled',
            Image),
        'bgr8')
    cv2.imwrite(f'{photo_num}.jpg', img)
    photo_num += 1


def detect_pub(cv_image, hsv):
    # детектируем пробои и проливы и выводим изображение в топики
    # а также обновляем информацию о пробоях и проливах
    global defect_pub, oil_pub
    
    # разлив
    black = cv2.inRange(hsv, *OIL_HSV)
    contours_blk, _ = cv2.findContours(
        black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    oil_area_px = max(map(cv2.contourArea, contours_blk), default=0)
    oil_area = convert_area(drone_height, oil_area_px)
    oil_img = cv_image.copy()
    # for i in range(len(contours_blk)):
    # cv2.drawContours(oil_img, contours_blk[i], -1, OIL_CONTOUR_COLOR, CONTOUR_WIDTH)
    cv2.drawContours(oil_img, contours_blk, -1, 
                     OIL_CONTOUR_COLOR, CONTOUR_WIDTH)
    oil_pub.publish(bridge.cv2_to_imgmsg(oil_img, 'bgr8'))
    
    # пробой
    black = cv2.inRange(hsv, *DEFECT_HSV)
    contours_blk, _ = cv2.findContours(
        black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    defect_img = cv_image.copy()
    # for i in range(len(contours_blk)):
    # cv2.drawContours(defect_img, contours_blk[i], -1, DEFECT_CONTOUR_COLOR, CONTOUR_WIDTH)
    cv2.drawContours(defect_img, contours_blk, -1, 
                     DEFECT_CONTOUR_COLOR, CONTOUR_WIDTH)
    defect_pub.publish(bridge.cv2_to_imgmsg(defect_img, 'bgr8'))

    if contours_blk:
        # если есть пробои, берём максимальный по площади (остальные могут быть
        # шумом)
        max_cnt = max(contours_blk, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) > MIN_DEFECT_AREA:
            # если размер пробоя меньше минимального, обновляем пробои
            screen_pos = get_contour_center(max_cnt)
            telem = get_telemetry(frame_id='aruco_map')
            map_pos = telem.x, telem.y
            update_defect_pos(screen_pos, map_pos, oil_area)


def rotate_to_line(black):
    h, w = SIZE
    xc, yc = 0, 0
    k = 0
    fi = 0
    for i in range(0, h):  # нахождение крайних точек по краям
        if (black[i][0] == 255):
            yc += i
            xc += 0
            k += 1
        if (black[i][w - 1] == 255):
            yc += i
            xc += w - 1
            k += 1
    for i in range(0, w):  # нахождение крайних точек с верху и с низу
        if (black[0][i] == 255):
            xc += i
            yc += 0
            k += 1
        if (black[h - 1][i] == 255):
            xc += i
            yc += h - 1
            k += 1
    if k == 0:
        print('k == 0')
        return
    # определение среднего положения крайних точек
    xc /= k  
    yc /= k
    print(xc, yc)
    xc -= w / 2
    print(xc, yc)  # определение угла поворота
    if (xc == 0):  # вырожденные случаи
        fi = (0 if yc == 0 else math.pi)
    elif (abs(xc) == w / 2 and yc==h/2):
        fi = (-math.pi / 2 if xc > 0 else math.pi / 2)
    else:
        if (yc > h / 2):  # 3 и 4 четверть
            fi = (math.atan2(yc - h / 2, abs(xc)) +
                  math.pi / 2) * (-1 if xc > 0 else 1)
        else:  # 1 и 2 четверть
            fi = (math.atan2(abs(xc), h / 2 - yc)) * (-1 if xc > 0 else 1)
    print('fi:', fi)  # поворот на нужный угол
    navigate(yaw=fi, z=0, frame_id='body')
    rospy.sleep(4)
    print('Rotated')


def convert_area(h, a):
    return 4 * h ** 2 * a * T / (320 * 240)


def line_callback(data):
    # полёт по линии
    global following_line, rotated
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  # OpenCV image
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    black = cv2.inRange(hsv, *LINE_HSV)
    contours_blk, _ = cv2.findContours(
        black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours_blk.sort(key=cv2.minAreaRect)
    contours_blk.sort(key=len)
    # выводим в топик контуры на картинке для дебага
    line_img = cv_image.copy()
    for i in range(len(contours_blk)):
        rect = cv2.minAreaRect(contours_blk[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(line_img, [box], 0, (0, 0, i * 21), 2)
        cv2.drawContours(line_img, contours_blk[i], -1, (i * 21, 0, 0), 1)
    if len(contours_blk) > 0:
        cnt = contours_blk[-1]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(line_img, [box], 0, (0, 255, 0), 2)
    line_pub.publish(bridge.cv2_to_imgmsg(line_img, 'bgr8'))
    # проверяем флаг
    if not following_line:
        return
    if not rotated:
        # поворачиваемся в сторону следования линии
        # correct_angle(black)
        print('angle0:', get_angle(black))
        rotate_to_line(black)
        rotated = True
    # детектим пробои и разливы
    detect_pub(cv_image, hsv)
    # есть контуры - летим
    if len(contours_blk) > 0:
        cnt = contours_blk[-1]
        if cv2.contourArea(cnt) > MIN_LINE_AREA:
            rect = cv2.minAreaRect(cnt)
            (x_min, y_min), (w_min, h_min), angle = rect
            if angle < -45:
                angle = 90 + angle
            if w_min < h_min and angle > 0:
                angle = (90 - angle) * -1
            if w_min > h_min and angle < 0:
                angle = 90 + angle
            center = cv_image.shape[1] / 2
            error = x_min - center
            # print(round(angle, 2), error)
            # drone_height = get_telemetry(frame_id='aruco_map').z
            # print('laser height:', drone_height, ' aruco height:', get_telemetry(frame_id='aruco_map').z)
            set_velocity(vx=0.01,
                         vy=error * (-0.004),
                         vz=(LINE_FLY_HEIGHT - drone_height) * 0.3,
                         yaw=float('nan'),
                         yaw_rate=angle * (-0.008),
                         frame_id='body')
            return
    # нет контуров - прекращаем полёт
    following_line = False
    set_velocity(vx=0, vy=0, vz=min((LINE_FLY_HEIGHT - drone_height) * 0.3, 0), frame_id='body')


def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.4,
                  yaw_rate=0, frame_id='', auto_arm=False, tolerance=0.2):
    navigate(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        speed=speed,
        frame_id=frame_id,
        auto_arm=auto_arm,
        yaw_rate=yaw_rate)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)


def correct_angle(black):
    angle = get_angle(black)
    print('Rotate angle:', angle)
    navigate(yaw=angle, frame_id='body')
    rospy.sleep(3)
    print('Rotated')


def get_angle(black):
    h, w = SIZE
    left = 0
    right = 0
    left_k = 0
    right_k = 0
    for i in range(0, w // 2):
        if black[0][i] == 255:
            left += math.atan(2 * (w // 2 - i) / h)
            left_k += 1
        if black[-1][i] == 255:
            left += math.pi - math.atan(2 * (w // 2 - i) / h)
            left_k += 1
    for i in range(w // 2, w):
        if black[0][i] == 255:
            right += math.atan(2 * (i - w // 2) / h)
            right_k += 1
        if black[-1][i] == 255:
            right += math.pi - math.atan(2 * (i - w // 2) / h)
            right_k += 1
    for i in range(0, h // 2):
        if black[i][0] == 255:
            left += math.pi / 2 - math.atan(2 * (h // 2 - i) / w)
            left_k += 1
        if black[i][-1] == 255:
            right += math.pi / 2 - math.atan(2 * (h // 2 - i) / w)
            right_k += 1
    for i in range(h // 2, h):
        if black[i][0] == 255:
            left += math.pi / 2 + math.atan(2 * (h // 2 - i) / w)
            left_k += 1
        if black[i][-1] == 255:
            right += math.pi / 2 + math.atan(2 * (h // 2 - i) / w)
            right_k += 1
    if left > right:
        return left / left_k
    return -right / right_k


def find_oil_callback(data):
    # Поиск пролива и публикация в топик red, для дебага
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
    img_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(img_hsv, (0, 40, 50), (25, 255, 255))
    b_moments = cv2.moments(red)
    b_pixel_area = b_moments["m00"] / 255
    # print(b_pixel_area)
    red_pub.publish(bridge.cv2_to_imgmsg(red, 'mono8'))


def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.4,
                  frame_id='', auto_arm=False, tolerance=0.2):
    navigate(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        speed=speed,
        frame_id=frame_id,
        auto_arm=auto_arm)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)


def find_qr_callback(data):
    # Считывание данных с QR кода при взлёте
    global qr_data
    if qr_data is not None:
        # QR код уже найден, ничего не делаем
        return
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  # OpenCV image
    barcodes = pyzbar.decode(cv_image)
    for barcode in barcodes:
        qr_data = barcode.data.decode("utf-8")
        return


def qr_land(data):
    global lost_qr
    if not landing or lost_qr:
        return
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  # OpenCV image
    barcodes = pyzbar.decode(cv_image)
    for barcode in barcodes:
        if drone_height < 0.2:
            break
        (x, y, w, h) = barcode.rect  # left, top, width, height
        xc = x + w / 2
        yc = y + h / 2
        # cv_image.shape = height, width

        # центр кадра
        img_x, = cv_image.shape[1] / 2
        img_y = cv_image.shape[0] / 2

        error_x = xc - img_x
        error_y = yc - img_y
        set_velocity(
            vx=error_y * -QR_K,
            vy=error_x * -QR_K,
            # vz=-0.1,
            vz=(0 if error_x ** 2 + error_y ** 2 < 400 else -0.1),
            frame_id='body')
        return
    lost_qr = True
    land()
    
def do_withdrawal(x, y):
    # Забор воды
    navigate_wait(x=x, y=y, z=1, frame_id='aruco_map')
    navigate_wait(x=x, y=y, z=0.6, frame_id='aruco_map')
    # Создаём видимость работы
    rospy.sleep(5)
    print('Successful water withdrawal')
    navigate_wait(x=x, y=y, z=2, frame_id='aruco_map')


# Подписываемся на топики
rospy.Subscriber('rangefinder/range', Range, range_callback)
rospy.Subscriber('main_camera/image_raw_throttled', Image, line_callback)
rospy.Subscriber('main_camera/image_raw_throttled', Image, find_oil_callback)
rospy.Subscriber('main_camera/image_raw_throttled', Image, find_qr_callback)
rospy.Subscriber('main_camera/image_raw_throttled', Image, qr_land)


with suppress(KeyboardInterrupt):
    # Взлёт
    print('Starting')
    navigate_wait(z=0.5, frame_id='body', auto_arm=True)
    print('Searching QR')
    # Поиск QR кода (логика поиска в find_qr_callback)
    while qr_data is None:
        rospy.sleep(0.1)
    (line_x, line_y), (lake_x, lake_y) = map(lambda x: map(float, x.split()), qr_data.split('\n'))
    print('QR data:')
    print(f'Line pos: {line_x} {line_y}')
    print(f'Lake pos: {lake_x} {lake_y}')
    # Сохраняем позицию взлёта
    start = get_telemetry(frame_id='aruco_map')
    # take_photo()
    # Делаем забор воды
    print('Doing withdrawal')
    do_withdrawal(lake_x, lake_y)
    # Летим к началу трубопровода и снижаемся
    navigate_wait(x=line_x, y=line_y, z=1, frame_id='aruco_map')
    navigate_wait(x=line_x, y=line_y, z=LINE_FLY_HEIGHT, frame_id='aruco_map', speed=0.3, tolerance=0.08)
    rospy.sleep(2)
    # navigate_wait(
        # x=line_x,
        # y=line_y,
        # z=LINE_FLY_HEIGHT,
        # frame_id='aruco_map',
        # yaw=3.14)
    # take_photo()
    
    print('Following line')
    # Летим по линии (логика полёта в line_callback)
    following_line = True
    while following_line:
        rospy.sleep(0.2)
    print('Returning')
    # Возвращаемся
    navigate_wait(x=start.x, y=start.y, z=2, frame_id='aruco_map')
    navigate_wait(
        x=start.x,
        y=start.y,
        z=1,
        frame_id='aruco_map',
        speed=0.3,
        tolerance=0.1)
    # take_photo()
    # take_photo()
    # Садимся по QR коду, пытаясь удержать его в центре кадра (логика посадки в qr_land)
    landing = True
    print('Landing to QR')
    while not lost_qr:
        rospy.sleep(0.2)
# Выводим отчёт, даже если коптер остановлен
print(f'Navigation area x={line_x}, y={line_y}')
print(f'Lake center x={lake_x}, y={lake_y}')
for i in range(len(defects)):
    print(
        f'{i + 1}. x={defects[i].map_pos[0]}, y={defects[i].map_pos[1]}, S={defects[i].oil_area}')
