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
    # print(drone_height)

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
    black = cv2.inRange(hsv, (30,80,160), (40,200,250))
    contours_blk, _  = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    defect_img = cv_image.copy()
    for i in range(len(contours_blk)):
        cv2.drawContours(defect_img,contours_blk[i],-1,(180, 105, 255),3)
    defect_pub.publish(bridge.cv2_to_imgmsg(defect_img, 'bgr8'))
    
    # разлив
    black = cv2.inRange(hsv, (0, 80, 130), (20, 180, 180))
    contours_blk, _  = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    oil_img = cv_image.copy()
    for i in range(len(contours_blk)):
        cv2.drawContours(oil_img,contours_blk[i],-1,(255, 0, 0),3)
    oil_pub.publish(bridge.cv2_to_imgmsg(oil_img, 'bgr8'))
    



def line(data):
    # Полёт по линии
    global isp
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')  # OpenCV image
    
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    detect_pub(cv_image, hsv)
    black = cv2.inRange(hsv, (30,80,160), (40,200,250))
    contours_blk, _  = cv2.findContours(black.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blk.sort(key=cv2.minAreaRect)
    # Выводим в топик контуры на картинке для дебага
    for i in range(len(contours_blk)):
        rect = cv2.minAreaRect(contours_blk[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(cv_image,[box],0,(0,0,i*21),2)
        cv2.drawContours(cv_image,contours_blk[i],-1,(i*21,0,0),1)
    if (len(contours_blk)>0):
        cnt =contours_blk[-1]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect) 
        box = np.int0(box)
        cv2.drawContours(cv_image,[box],0,(0,255,0),2)
    polet_pub.publish(bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
    # Проверяем флаг
    if isp==0:
        return
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
            print(round(angle, 2), error)
            # drone_height = get_telemetry(frame_id='aruco_map').z
            set_velocity(vx=0.1, vy=error * (-0.01), vz=(1 - drone_height) * 0.3, yaw=float('nan'), yaw_rate=angle * (-0.008), frame_id='body')
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
    red = cv2.inRange(img_hsv, (0, 158, 50), (3, 255, 255))
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
navigate_wait(x=qr_x, y=qr_y, z=1, frame_id='aruco_map')
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
QR_K = 0.01  # коэффициент  
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