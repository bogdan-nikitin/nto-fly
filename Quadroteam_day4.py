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


# Класс данных о пробое
@dataclass
class Defect:
    screen_dist: float  # расстояние до центра изображения (в пикселях)
    map_pos: (float, float)  # позиция на aruco карте


defects = []  # найденные дефекты

# разрещение камеры дрона
SCREEN = 240, 320  # HEIGHT, WIDTH

# минимальное расстояние между дефектами
DEFECT_MIN_DIST = 0.3

# минимальная площадь дефекта в пикселях
MIN_DEFECT_AREA = 100


def get_contour_center(cnt):
    moment = cv2.moments(cnt)
    x = int(moment['m10']/moment['m00'])
    y = int(moment['m01']/moment['m00'])
    return x, y


def get_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def update_defect_pos(screen_pos, map_pos):
    # функция для вывода и сохранения координат разливов
    global defects
    screen_center = SCREEN[0] / 2, SCREEN[1] / 2
    # считаем расстояние от позиции разлива на изображении до центра изображения
    screen_dist = get_distance(screen_center, screen_pos)
    
    # проверяем, не находили ли мы этот дефект ранее
    for defect in defects:
        # если разница в позиции меньше минимальной, то мы находили дефект ранее
        if get_distance(map_pos, defect.map_pos) < DEFECT_MIN_DIST:
            # если расстояние до центра изображения меньше, обновляем позицию на более точную
            if screen_dist < defect.screen_dist:
                defect.screen_dist = screen_dist
                defect.map_pos = map_pos
            return
    # если такого пробоя не было, добавляем в список
    defects.append(Defect(screen_dist, map_pos))
    print(f'defect: {map_pos[0]} {map_pos[1]}')


# Номер последней фотки
photo_num = 1

bridge = CvBridge()
# Высота дрона над землёй
drone_height = 0 


def range_callback(msg):
    # Обработка новых данных с дальномера
    global drone_height
    if msg.range != float('inf') and msg.range != float('nan'):
        drone_height = msg.range

rospy.Subscriber('rangefinder/range', Range, range_callback)

def take_photo():
    # Снятие фото с камеры для дебага
    global photo_num
    img = bridge.imgmsg_to_cv2(rospy.wait_for_message('main_camera/image_raw_throttled', Image), 'bgr8')
    cv2.imwrite(f'{photo_num}.jpg', img)
    photo_num += 1

rospy.init_node('flight')

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)

# Флаг. 1 - летим над линией, 0 - ничего не делаем
isp=0  
polet_pub = rospy.Publisher('polet', Image, queue_size=1)
defect_pub = rospy.Publisher('defect_detect', Image, queue_size=1)
oil_pub = rospy.Publisher('oil_detect', Image, queue_size=1)


def detect_pub(cv_image, hsv):
    # Детектируем пробои и проливы и выводим изображение в топики
    global defect_pub, oil_pub
    # пробой
    black = cv2.inRange(hsv, (35,200,20), (60,255,50))
    contours_blk, _  = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    defect_img = cv_image.copy()
    for i in range(len(contours_blk)):
        cv2.drawContours(defect_img,contours_blk[i],-1,(180, 105, 255),3)
    defect_pub.publish(bridge.cv2_to_imgmsg(defect_img, 'bgr8'))
    if contours_blk:
        max_cnt = max(contours_blk, key=cv2.contourArea)
        if cv2.contourArea(max_cnt) > MIN_DEFECT_AREA:
            # если размер пробоя меньше минимального, обновляем пробои
            screen_pos = get_contour_center(max_cnt)
            telem = get_telemetry(frame_id='aruco_map')
            map_pos = telem.x, telem.y
            update_defect_pos(screen_pos, map_pos)
        
        
    
    # разлив
    black = cv2.inRange(hsv, (0, 80, 130), (20, 180, 180))
    contours_blk, _  = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    oil_img = cv_image.copy()
    for i in range(len(contours_blk)):
        cv2.drawContours(oil_img,contours_blk[i],-1,(255, 0, 0),3)
    oil_pub.publish(bridge.cv2_to_imgmsg(oil_img, 'bgr8'))
    
def rotate_to_line(cv_image):
    # расчёт угла поворота на старте линии и поворот на этот угол
    img = cv_image.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    black = cv2.inRange(hsv, (0, 100, 200), (50,255,255))
    h, w, _ = img.shape
    xc,yc=0,0
    k=0
    fi=0
    for i in range(0,h):
        if black[i][0] or black[i][w-1]:
            print(black[i][0], black[i][w - 1])
        if (black[i][0]==255):
            yc+=i
            k+=1
        if (black[i][w-1]==255):
            yc+=i
            k+=1
    for i in range(0,w):
        if (black[0][i]==255):
            xc+=i
            k+=1
        if (black[h-1][i]==255):
            xc+=i
            k+=1

    xc/=k
    yc/=k
    print(xc,yc)
    xc-=w/2
    print(xc,yc)
    if (xc==0):
        fi=(0 if yc==0 else math.pi)
    elif (abs(xc)==w/2):
        fi=(-math.pi/2 if yc>h/2 else math.pi/2)
    else:
        if (yc>h/2):
            fi=(math.atan2(yc-h/2,abs(xc))+math.pi/2)*(-1 if xc>0 else 1)
        else:
            fi=(math.atan2(abs(xc),h/2-yc))*(-1 if xc>0 else 1)
    print(fi)
    navigate_wait(yaw=-fi, z=0, frame_id='body')

rotated = False

def line(data):
    # Полёт по линии
    global isp, rotated
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  # OpenCV image
    
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    black = cv2.inRange(hsv, (0, 80, 130), (20, 180, 180))
    contours_blk, _  = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_blk.sort(key=cv2.minAreaRect)
    contours_blk.sort(key=len)
    # Выводим в топик контуры на картинке для дебага
    line_img = cv_image.copy()
    for i in range(len(contours_blk)):
        rect = cv2.minAreaRect(contours_blk[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(line_img,[box],0,(0,0,i*21),2)
        cv2.drawContours(line_img,contours_blk[i],-1,(i*21,0,0),1)
    if (len(contours_blk)>0):
        cnt =contours_blk[-1]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect) 
        box = np.int0(box)
        cv2.drawContours(line_img,[box],0,(0,255,0),2)
    polet_pub.publish(bridge.cv2_to_imgmsg(line_img, 'bgr8'))
    # Проверяем флаг
    if isp==0:
        return
    if not rotated:
        rotate_to_line(cv_image)
        rotated = True
    detect_pub(cv_image, hsv)
    # Есть контуры - летим
    if len(contours_blk) > 0:
        cnt = contours_blk[-1]
        if cv2.contourArea(cnt) > 300:
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
            set_velocity(vx=0.05, vy=error * (-0.01), vz=(1 - drone_height) * 0.3, yaw=float('nan'), yaw_rate=angle * (-0.008), frame_id='body')
    # Нет контуров - прекращаем полёт
    else:
        isp=0
rospy.Subscriber('main_camera/image_raw_throttled', Image, line)


def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.4, frame_id='', auto_arm=False, tolerance=0.2):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)

red_pub = rospy.Publisher('red', Image, queue_size=1)
		
def image_callback1(data):
    # Поиск пролива и публикация в топик red, для дебага
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  
    img_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(img_hsv, (0, 40, 50), (25, 255, 255))
    b_moments = cv2.moments(red)
    b_pixel_area = b_moments["m00"]/255 
    # print(b_pixel_area)   
    red_pub.publish(bridge.cv2_to_imgmsg(red, 'mono8'))
rospy.Subscriber('main_camera/image_raw_throttled', Image, image_callback1)

rospy.Subscriber('main_camera/image_raw_throttled', Image, line)


def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.4, frame_id='', auto_arm=False, tolerance=0.2):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)

red_pub = rospy.Publisher('red', Image, queue_size=1)
		
def image_callback1(data):
    # Поиск пролива и публикация в топик red, для дебага
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  
    img_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(img_hsv, (0, 40, 50), (25, 255, 255))
    b_moments = cv2.moments(red)
    b_pixel_area = b_moments["m00"]/255 
    # print(b_pixel_area)   
    red_pub.publish(bridge.cv2_to_imgmsg(red, 'mono8'))
rospy.Subscriber('main_camera/image_raw_throttled', Image, image_callback1)


qr_x, qr_y = None, None


def image_callback(data):
    # Считывание данных с QR кода при взлёте
    global qr_x, qr_y
    if (qr_x, qr_y) != (None, None):
        # QR код уже найден, ничего не делаем
        return
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  # OpenCV image
    barcodes = pyzbar.decode(cv_image)
    for barcode in barcodes:
        b_data = barcode.data.decode("utf-8")
        qr_x, qr_y = tuple(map(float, b_data.split()))
        print(qr_x, qr_y)
        return


rospy.Subscriber('main_camera/image_raw_throttled', Image, image_callback, queue_size=1)
# Взлёт
navigate_wait(z=1, frame_id='body', auto_arm=True, speed=0.2)
# Поиск QR кода
while (qr_x, qr_y) == (None, None):
    rospy.sleep(0.1)
# Сохраняем позицию взлёта
start = get_telemetry(frame_id='aruco_map')
take_photo()
# Летим к началу трубопровода и снижаемся
navigate_wait(x=qr_x, y=qr_y, z=2, frame_id='aruco_map')
navigate_wait(x=qr_x, y=qr_y, z=1, frame_id='aruco_map', yaw=3.14)
take_photo()

# print('Following line')
# Летим по линии
isp = 1
while isp==1:
    rospy.sleep(0.2)
# print('Returning')
# Возвращаемся
navigate_wait(x=start.x, y=start.y, z=2, frame_id='aruco_map')
take_photo()
navigate_wait(x=start.x, y=start.y, z=1, frame_id='aruco_map', tolerance=0.05, speed=0.2)
take_photo()

# Садимся по QR коду, пытаясь удержать его в центре кадра
lost_qr = False
QR_K = 0.005  # коэффициент  
def qr_land(data):
    global lost_qr
    if lost_qr:
        return
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  # OpenCV image
    barcodes = pyzbar.decode(cv_image)
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect  # left, top, width, height
        xc = x + w / 2
        yc = y + h / 2
        # cv_image.shape = height, width
        img_x, img_y = cv_image.shape[1] / 2, cv_image.shape[0] / 2  # центр кадра
        error_x = xc - img_x
        error_y = yc - img_y
        set_velocity(vx=error_y * -QR_K, vy=error_x * -QR_K, vz=-0.1, frame_id='body')
        return
    lost_qr = True
    land()
rospy.Subscriber('main_camera/image_raw_throttled', Image, qr_land)
# print('Landing to QR')
while not lost_qr:
    rospy.sleep(0.2)
print(f'Navigation area x={qr_x}, y={qr_y}')
