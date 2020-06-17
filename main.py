
# coding=utf-8
import os
import sys
from random import random

import cv2
from hyperlpr import *  # this is new plate recognition function package
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QStyle
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex, QMutexLocker, Qt
from UI_file import *
import numpy as np
import time
import threading
from queue import Queue
import xlwt

CURPATH = os.path.dirname(os.path.realpath(__file__))

plate_q = Queue()  # used to pass the result of plate recognition
type_q = Queue()  # same as above
type_confidence_q = Queue()
plate_confidence_q = Queue()
im_q = Queue()
img_car_q = Queue()
time_q = Queue()

even_yolo = threading.Event()  # to synchronize between tasks
even_model = threading.Event()

even_license = threading.Event()


##########################################################################################
def similar(str1, str2):
    str1 = str1 + ' ' * (len(str2) - len(str1))
    str2 = str2 + ' ' * (len(str1) - len(str2))
    return sum(1 if i == j else 0
               for i, j in zip(str1, str2)) / float(len(str1))


def compute_IOU(rec1, rec2):
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        # S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / S2


def transfer_time_format(str):
    if "_" in str:
        _ = str.split("_")
        return _[0] + "时" + _[1] + "分" + _[2] + "秒" + _[3] + "年" + _[4] + "月" + _[5] + "日"
    else:
        return str.replace("年", "_").replace("月", "_").replace("日", "").replace("时", "_").replace("分", "_").replace("秒",
                                                                                                                    "_")


# https://www.jianshu.com/p/7b6a80faf33f
class MODEL_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.model_thread_running = True

    def close(self):
        pass


class Car:
    def __init__(self, c_id, c_x, c_y, derection, c_count):
        self.c_id = c_id
        self.c_x = c_x
        self.c_y = c_y
        self.derection = derection
        self.c_count = c_count

    def updateCoords(self, x, y):
        self.c_x = x
        self.c_y = y


class YOLO_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.yolo_thread_running = True
        # self.daemon = True
        weightsPath = os.path.join(CURPATH, 'lib\yolo\yolov3.weights')
        configPath = os.path.join(CURPATH, 'lib\yolo\yolov3.cfg')
        labelsPath = os.path.join(CURPATH, 'lib\yolo\coco.names')
        self.LABELS = open(labelsPath).read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.BOUNDING = [0.02, 0.98, 0.300, 0.700]
        self.object_x = 0
        self.object_y = 0
        self.object_w = 0
        self.object_h = 0
        self.IOU = 0.8

        self.car_q = []
        self.pid = 1
        self.count_up = 0
        self.count_down = 0

    def run(self):
        print("运行一次")
        global what_pt_want
        global yoloimg
        print('yolo done')
        while self.yolo_thread_running:
            even_yolo.wait()

            if not type_q.empty():
                type_q.get()

            if not plate_q.empty():
                plate_q.get()

            if not type_confidence_q.empty():
                type_confidence_q.get()

            if not plate_confidence_q.empty():
                plate_confidence_q.get()

            ###############################################################

            classIDs = []
            if im_q.empty():
                time.sleep(0.5)
                continue
            # img_o = im_q.get()
            if not im_q.empty():
                img = cv2.cvtColor(im_q.get(), cv2.COLOR_RGB2BGR)
                (H, W) = img.shape[:2]
                ln = self.net.getLayerNames()
                ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

                blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                self.net.setInput(blob)
                layerOutputs = self.net.forward(ln)
                boxes = []
                confidences = []

                for output in layerOutputs:
                    # print("一个")
                    # 对每个检测进行循环
                    for detection in output:
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        if self.LABELS[classID] not in ('truck', 'car', 'bus', 'train', 'bicycle'):  # 'motorbike'
                            continue
                        confidence = scores[classID]
                        if confidence > 0.5:
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
                            # 边框的左上角
                            if centerX < 0 or centerY < 0 or width < 0 or height < 0:
                                continue
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)

                            # showme = img_o[x:x+width,y:y+ height]
                            # cv2.imshow("showme",showme)

                # 极大值抑制
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
                if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        cx = x + w / 2
                        cy = y + h / 2

                        rec_ = [W * self.BOUNDING[0], H * self.BOUNDING[2], W * self.BOUNDING[1], H * self.BOUNDING[3]]
                        rec_car = [x, y, x + w, y + h]
                        # cv2.rectangle(img, (int(W*self.BOUNDING[0]), int(H*self.BOUNDING[2])), (int(W*self.BOUNDING[1]), int(H*self.BOUNDING[3])), (255, 0, 0), 2)

                        # cv2.imshow("show",img)
                        # img_c = img[y:y + h, x:x + w]
                        # cv2.imshow("mycar",img_c)

                        IOU = compute_IOU(rec_, rec_car)
                        print("重叠度" + str(IOU))
                        expression2 = IOU > self.IOU
                        # expression2 =True
                        if expression2:
                            new = True
                            for j in self.car_q:
                                if abs(cx - j.c_x) < w and abs(cy - j.c_y) < h:
                                    new = False
                                    print("同一辆车")
                                    # 防止开头
                                    #if j.c_id > 2:
                                    if cy > j.c_y and random() > 0.5:
                                        #i.derection = 'down'
                                        self.count_down += 1
                                    if cy < j.c_y and random() > 0.5:
                                        #i.derection = 'up'
                                        self.count_up += 1
                                    self.car_q.remove(j)
                                    break
                                    # self.pid =self.pid + 1
                                    # p = Car(self.pid, cx, cy, 'unknow', True)
                                    # self.car_q.append(p)


                            if new == True:
                                type_q.put(self.LABELS[classIDs[i]])
                                type_confidence_q.put(str(confidences[i]))

                                nd.settable(False, self.LABELS[classIDs[i]], str(confidences[i]), False, False)
                                img_crop = img[y:y + h, x:x + w]
                                # cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),(255, 0, 0), 2)
                                ################################
                                save_time = time.strftime('%H{h}%M{f}%S{s}%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日',
                                                                                                   h='时',
                                                                                                   f='分', s='秒')
                                filepath = os.path.join(CURPATH,
                                                        'archives\{}.jpg'.format(transfer_time_format(save_time)))
                                print(filepath)
                                try:
                                    flag_ = cv2.imwrite(filepath, img_crop)  # save vehicle image by name of time
                                except:
                                    print('图片保存失败')
                                nd.settable(save_time, False, False, False, False)
                                #########################
                                img_car_q.put(img_crop)
                                p = Car(self.pid, cx, cy, 'unknow', False)
                                self.pid = self.pid + 1
                                self.car_q.append(p)
                                even_license.set()
                                break
                    even_yolo.clear()
                    # break

    def close(self):
        self.yolo_thread_running = False
        print('closing yolo session')


class LICENSE_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.license_thread_running = True
        # self.daemon = True

    def run(self):
        license_warmup = np.zeros([400, 400, 3], np.uint8)
        del license_warmup
        while self.license_thread_running:
            even_license.wait()
            # if not img_car_q.empty():
            # if True:
            image = img_car_q.get()
            # if image:
            try:
                tem = HyperLPR_plate_recognition(image)
            except:
                print("识别失败")
            print(tem)
            plate_into = "".join('%s' % id for id in tem)
            if 6 <= len(plate_into.split(',')[0].replace('[', '').replace('\'', '')) <= 8:
                nd.settable(False, False, False, plate_into.split(',')[0].replace('[', '').replace('\'', ''), False)
                nd.settable(False, False, False, False, plate_into.split(',')[1])
            else:
                nd.settable(False, False, False, '', False)
                nd.settable(False, False, False, False, '')

    def close(self):
        self.license_thread_running = False
        print('closing license')


class Car:
    def __init__(self, c_id, c_x, c_y, derection, c_count):
        self.c_id = c_id
        self.c_x = c_x
        self.c_y = c_y
        self.derection = derection
        self.c_count = c_count

    def updateCoords(self, x, y):
        self.c_x = x
        self.c_y = y


class mywindow(Ui_Dialog):
    global what_pt_want
    status = 0
    video_type = 0
    TYPE_VIDEO = 0
    TYPE_CAMERA = 1
    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2
    CAMID = 'qq'
    BEFORE = '0'
    global pt_video_counter

    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        ############################
        self.yolo_thread = YOLO_thread()
        self.license_thread = LICENSE_thread()
        self.model_thread = MODEL_thread()
        self.yolo_thread.start()
        self.license_thread.start()
        self.model_thread.start()
        ###########################
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.line_counter = 0

        self.table_dict = {}
        self.table_dict['time'] = Queue()
        self.table_dict['type'] = Queue()
        self.table_dict['type_confidence'] = Queue()

        self.table_dict['plate'] = Queue()
        self.table_dict['plate_confidence'] = Queue()

        self.timer = VideoTimer()
        self.timer.timeSignal.signal[str].connect(
            self.show_video_images)  # once timer emit a signal,run show_video_images()
        self.INTERVAL = 0.5

        self.count_up = 0
        self.count_down = 0
        self.cars = []

        # 模型路径
        # self.model_bin = r"F:\ssd\MobileNetSSD_deploy.caffemodel"
        # self.config_text = r"F:\ssd\MobileNetSSD_deploy.prototxt"
        # # 类别信息
        # self.objName = ["background",
        #                 "aeroplane", "bicycle", "bird", "boat",
        #                 "bottle", "bus", "car", "cat", "chair",
        #                 "cow", "diningtable", "dog", "horse",
        #                 "motorbike", "person", "pottedplant",
        #                 "sheep", "sofa", "train", "tvmonitor"]
        #
        # # 加载模型
        # self.net = cv2.dnn.readNetFromCaffe(self.config_text, self.model_bin)

        self.cars = []
        self.count_up = 0
        self.count_down = 0
        self.pid = 1

    def openimage(self):
        global pt_video_counter
        self.video_type = self.TYPE_VIDEO
        self.reset()
        if 'self.playCapture' in locals() or 'self.playCapture' in globals():
            self.playCapture.release()
        imgName, imgType = QFileDialog.getOpenFileName(self, "open the image", "",
                                                       " All Files (*);;*.asf;;*.mp4;;*.mpg;;*.avi")
        global what_pt_want
        what_pt_want = imgName
        if 'what_pt_want' in locals() or 'what_pt_want' in globals():
            pass
            if what_pt_want == "":
                return
            else:
                self.pushButton.setEnabled(True)
                self.playCapture = cv2.VideoCapture()
                pt_video_counter = 1
                self.playCapture.open(what_pt_want)
                fps = self.playCapture.get(cv2.CAP_PROP_FPS)  # used to be cv2.CAP_PROP_FPS
                if fps == 0:
                    QMessageBox.warning(self, 'error', 'fps不能为0')
                    return
                else:
                    self.timer.set_fps(fps)

                self.timer.start()
                self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                self.status = (self.STATUS_PLAYING, self.STATUS_PAUSE, self.STATUS_PLAYING)[self.status]
        else:
            return

    def settable_function(self, type_signal, result):
        self.table_dict[type_signal].put(result)
        if not (self.table_dict['time'].empty() or self.table_dict['type'].empty() or self.table_dict[
            'type_confidence'].empty()
                or self.table_dict['plate'].empty() or self.table_dict['plate_confidence'].empty()):

            time = self.table_dict['time'].get()
            type = self.table_dict['type'].get()

            type_confidence = self.table_dict['type_confidence'].get()

            plate = self.table_dict['plate'].get()
            plate_confidence = self.table_dict['plate_confidence'].get()

            print("上一次" + self.BEFORE)
            print("这次" + plate.strip())
            # if plate.strip() =="NONE" or self.BEFORE != plate.strip():
            if similar(self.BEFORE, plate.strip()) <= 0.5 and plate.strip() != "":
                self.tableWidget.insertRow(self.line_counter)
                self.tableWidget.setItem(self.line_counter, 0, QTableWidgetItem(time))
                self.tableWidget.setItem(self.line_counter, 1, QTableWidgetItem(type))
                self.tableWidget.setItem(self.line_counter, 2, QTableWidgetItem(type_confidence))
                self.tableWidget.setItem(self.line_counter, 3, QTableWidgetItem(plate))
                self.tableWidget.setItem(self.line_counter, 4, QTableWidgetItem(plate_confidence))

                self.line_counter += 1
                self.tableWidget.verticalScrollBar().setValue(self.line_counter)
                self.BEFORE = plate.strip()

    def reset(self):
        self.timer.stopped = True
        if 'self.playCapture' in locals() or 'self.playCapture' in globals():
            self.playCapture.release()
        self.status = self.STATUS_INIT
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def show_video_images(self):
        '''it detected a car of interst in wanted region,after inference,run this function'''
        # if video is open successfully,read it
        if self.playCapture.isOpened():
            success, frame_o = self.playCapture.read()
            # ,frame is a ndarray,frame.shape index 0 and 1 stand for height and width
            if success:
                global pt_video_counter
                frame = cv2.cvtColor(frame_o, cv2.COLOR_BGR2RGB)

                h, w = frame.shape[:2]
                image = frame

                if (pt_video_counter % (int)(self.timer.frequent * self.INTERVAL) == 0):  # INTERVAL
                    self.line_up.setText(str(self.yolo_thread.count_up))
                    self.line_down.setText(str(self.yolo_thread.count_down))
                    im_q.put(frame)  # 6_16,modified
                    even_yolo.set()
                    # blobImage = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False)
                    # self.net.setInput(blobImage)
                    # cvOut = self.net.forward()
                    # #cv2.line(frame, (0, 700), (1920, 700), (0, 255, 0), 3)
                    # #cv2.line(frame, (0, 400), (1920, 400), (0, 255, 0), 3)
                    # #cv2.line(frame, (1000, 0), (1000, 1080), (0, 255, 0), 1)
                    # cv2.putText(frame, "Up:" + str(self.count_up), (100, 200), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                    # cv2.putText(frame, "Down:" + str(self.count_down), (100, 300), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                    # for detection in cvOut[0, 0, :, :]:
                    #     score = float(detection[2])
                    #     objIndex = int(detection[1])
                    #     # if score > 0.1 and (objName[objIndex] =='car' or objName[objIndex] =='bus'):
                    #     if score > 0.1 :
                    #         left = detection[3] * w
                    #         top = detection[4] * h
                    #         right = detection[5] * w
                    #         bottom = detection[6] * h
                    #         # print(left,right,top,bottom)
                    #         # 绘制
                    #         if 300 < int((top + bottom) / 2) < 800:
                    #             width = int(right - left)
                    #             height = int(bottom - top)
                    #             cx = int((left + right) / 2)
                    #             cy = int((top + bottom) / 2)
                    #             new = True
                    #             for i in self.cars:
                    #                 if abs(cx - i.c_x) < width and abs(cy - i.c_y) < height:  # 找到这辆车与上一帧中最近的车
                    #                     new = False
                    #                     # 最开始的想法，Y轴坐标比上一帧大方向向下，比上一帧小方向向上
                    #                     # if cy > i.c_y:
                    #                     #     i.derection = 'down'
                    #                     # if cy < i.c_y:
                    #                     #     i.derection = 'up'
                    #
                    #                     i.updateCoords(cx, cy)
                    #                     if i.c_y >= 750 and i.derection == 'down' and i.c_count == False:
                    #                         self.count_down += 1
                    #                         i.c_count = True
                    #                     if i.c_y < 350 and i.derection == 'up' and i.c_count == False:
                    #                         self.count_up += 1
                    #                         i.c_count = True
                    #                 if i.c_y >= 790 or i.c_y <= 310:
                    #                     self.cars.remove(i)
                    #
                    #             if new == True:
                    #                 p = Car(self.pid, cx, cy, 'unknow', False)
                    #                 # 判断方向，这里我是在没找到好的办法，只能从简，大家测试的时候需要修改
                    #                 if p.c_x > 1000:
                    #                     p.derection = 'up'
                    #                 else:
                    #                     p.derection = 'down'
                    #                 self.cars.append(p)
                    #                 self.pid += 1
                    #             # print(len(cars))
                    #             cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                    #             cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0),
                    #                          thickness=2)
                    #
                    #             cv2.putText(image, "score:%.2f, %s" % (score, self.objName[objIndex]),
                    #                        (int(left) - 10, int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8);

                #  显示
                temp_image = QImage(image.flatten(), w, h, QImage.Format_RGB888)
                temp_pixmap = QPixmap.fromImage(temp_image)
                temp_pixmap = temp_pixmap.scaled(self.graphicsView.width(), self.graphicsView.height())
                self.graphicsView.setPixmap(temp_pixmap)
                pt_video_counter += 1
            else:
                #time.sleep(3)
                print("read failed, no frame data")
                self.reset()
        else:
            print('end')
            self.reset()  # open file or capturing device error, init again

    def webcamera(self):
        self.reset()
        global pt_video_counter
        if not self.pushButton_3.isEnabled():
            return
        else:
            self.pushButton.setEnabled(True)
        self.playCapture = cv2.VideoCapture(self.CAMID.strip())
        fps = self.playCapture.get(cv2.CAP_PROP_FPS)  # used to be cv2.CAP_PROP_FPS
        if fps == 0:
            QMessageBox.warning(self, 'error', 'fps不能为0')
            return
        self.timer.set_fps(fps)
        self.video_type = self.TYPE_CAMERA
        self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        pt_video_counter = 1
        self.timer.start()
        self.status = (self.STATUS_PLAYING, self.STATUS_PAUSE, self.STATUS_PLAYING)[self.status]

    def inquiry(self):
        global pt_video_counter
        if not self.pushButton.isEnabled():
            return
        if self.status is self.STATUS_INIT:
            pass
        elif self.status is self.STATUS_PLAYING:
            self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.timer.stopped = True
            if self.video_type is self.TYPE_CAMERA:
                self.playCapture.release()
        elif self.status is self.STATUS_PAUSE:
            self.pushButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            if self.video_type is self.TYPE_CAMERA:
                self.webcamera()
            else:
                self.timer.start()
        self.status = (self.STATUS_PLAYING, self.STATUS_PAUSE, self.STATUS_PLAYING)[self.status]

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'warning', "你确定要退出吗?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for filepath in os.listdir(os.path.join(CURPATH, 'archives')):
                os.remove(os.path.join(os.path.join(CURPATH, 'archives'), filepath))
            event.accept()
            self.reset()
            self.yolo_thread.close()
            self.model_thread.close()
            # self.color_thread.close()
            self.license_thread.close()
            self.timer.wait()
            nd.wait()
            pass
            sys.exit(app.exec_())
        else:
            event.ignore()

    def display_table(self):
        line = self.tableWidget.currentRow()
        value = self.tableWidget.item(line, 0).text()
        image_file = os.path.join(CURPATH, 'archives\{}.jpg'.format(transfer_time_format(value)))
        print("打开" + image_file)
        if not os.path.exists(image_file):
            QMessageBox.about(self, 'error', '图片不存在')
            return
        result_image = QtGui.QPixmap(image_file).scaled(window.graphicsView_frame.width(),
                                                        window.graphicsView_frame.height())
        window.graphicsView_frame.setPixmap(result_image)

    def export_txt(self):
        save_path = QFileDialog.getSaveFileName(self, 'save file', CURPATH, 'txt(*txt)')
        save_path = save_path[0]
        if not save_path.endswith('.xls'):
            save_path = save_path + '.txt'
        try:
            predix = ' 保存时间：' + time.strftime('%H{h}%M{f}%S{s}%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日', h='时',
                                                                                       f='分', s='秒')
            if not os.path.exists(os.path.join(CURPATH, save_path)):
                predix = '%s' % ('时间') + '%45s' % ('类型') + '%20s' % ('置信度') + '%15s' % ('车牌') + '%25s' % (
                    '车牌置信度') + '\n' + predix
            with open(os.path.join(CURPATH, save_path), 'a+') as f:
                f.write(predix + '\n')
                for line in range(self.line_counter):
                    values = []
                    values.append('%20s' % (self.tableWidget.item(line, 0).text()))
                    values.append('%10s' % (self.tableWidget.item(line, 1).text()))
                    values.append('%10s' % (self.tableWidget.item(line, 2).text()))
                    values.append('%10s' % (self.tableWidget.item(line, 3).text()))
                    values.append('%5s' % (self.tableWidget.item(line, 4).text()))
                    f.write('      '.join(values) + '\n')
            QMessageBox.information(self, 'Great！', '已成功保存为%s' % (save_path))
        except Exception as e:
            print(repr(e))
            fname = 'error.txt'
            predix = ' 出错时间：' + time.strftime('%H{h}%M{f}%S{s}%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日', h='时',
                                                                                       f='分', s='秒')
            with open(fname, 'a+') as f:
                f.write('\n' + repr(e))
            QMessageBox.warning(self, 'save error!!', '  save error message to {}'.format(fname))

    def export(self):  # 先用.xls格式保存结果
        save_path = QFileDialog.getSaveFileName(self, 'save file', CURPATH, 'xls(*xls)')
        save_path = save_path[0]
        if not save_path:
            return
        try:
            workbook = xlwt.Workbook(encoding='utf-8')
            worksheet = workbook.add_sheet('My worksheet', cell_overwrite_ok=True)
            if not os.path.exists(os.path.join(CURPATH, save_path)):
                pass
            _ = 0
            for content_ in ['时间', '类型', '置信度', '车牌', '车牌置信度']:
                worksheet.write(0, _, label=content_)
                _ += 1
            for line in range(self.line_counter):
                for _ in range(5):
                    worksheet.write(line + 1, _, label=(self.tableWidget.item(line, _)).text())
                if not save_path.endswith('.xls'):
                    save_path = save_path + '.xls'
                workbook.save(save_path)
        except Exception as e:
            print(repr(e))
            QMessageBox.warning(self, '保存失败', repr(e))
            return
        QMessageBox.information(self, '保存成功', ' 已保存到 %s' % (str(save_path)))


class Communicate(QObject):
    signal = pyqtSignal(str)


class VideoTimer(QThread):
    def __init__(self, frequent=100):
        QThread.__init__(self)
        self.stopped = False
        self.frequent = frequent
        self.timeSignal = Communicate()
        self.mutex = QMutex()  # the lock between threads

    def run(self):  # method run just play the pix of the video one by one
        with QMutexLocker(self.mutex):
            self.stopped = False
        self.run_()

    def run_(self):
        while True:
            if self.stopped:
                return
            self.timeSignal.signal.emit("1")
            time.sleep(1 / self.frequent)

    def set_fps(self, fps):
        self.frequent = fps


class Network_daemon(QThread):
    '''daemon thread, function haha is used to display brand, license plate number,color and model'''
    trigger_table = pyqtSignal(str, str)

    def __int__(self):
        super(Network_daemon, self).__init__()

    def run(self):
        while True:
            time.sleep(5)
            # if not pinpai_img_q.empty():
            if not even_model.is_set():
                even_model.set()
        return

    def settable(self, timestr, typestr, type_confidencestr, platestr, plate_confidencestr):

        if timestr is not False:
            time_q.put(timestr)
            self.trigger_table.emit('time', time_q.get())

        if typestr is not False:
            type_q.put(typestr)
            self.trigger_table.emit('type', type_q.get())

        if type_confidencestr is not False:
            type_confidence_q.put(type_confidencestr)
            self.trigger_table.emit('type_confidence', type_confidence_q.get())

        if platestr is not False:
            plate_q.put(platestr)
            self.trigger_table.emit('plate', plate_q.get())

        if plate_confidencestr is not False:
            plate_confidence_q.put(plate_confidencestr)
            self.trigger_table.emit('plate_confidence', plate_confidence_q.get())

        # if speedstr is not False:
        #     speed_q.put(confidencestr)
        #     self.trigger_table.emit('speed', speed_q.get())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    global window
    window = mywindow()
    nd = Network_daemon()
    nd.trigger_table.connect(window.settable_function)
    nd.start()
    window.show()
    window.pushButton_2.setEnabled(True)
    sys.exit(app.exec_())

