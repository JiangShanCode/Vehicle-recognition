import os

import cv2 as cv
import time
CURPATH=os.path.dirname(os.path.realpath(__file__))
class Car:
    def __init__(self,c_id,c_x,c_y,derection,c_count):
        self.c_id = c_id
        self.c_x=c_x
        self.c_y=c_y
        self.derection = derection
        self.c_count = c_count

    def updateCoords(self,x,y):
        self.c_x= x
        self.c_y=y

count_up = 0
count_down = 0
cars = []
pid = 1
# 模型路径
model_bin = r"F:\ssd\MobileNetSSD_deploy.caffemodel"
config_text = r"F:\ssd\MobileNetSSD_deploy.prototxt"
# 类别信息
objName = ["background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor"]

# 加载模型
net = cv.dnn.readNetFromCaffe(config_text, model_bin)
# 读取测试图片
vc = cv.VideoCapture("C:\\Users\\江山\Desktop\\testvideo\\video-02.mp4")


def transfer_time_format(str):
    if "_" in str:
        _ = str.split("_")
        return _[0] + "时" + _[1] + "分" + _[2] + "秒" + _[3] + "年" + _[4] + "月" + _[5] + "日"
    else:
        # print(str.replace("年","_").replace("月","_").replace("日","").replace("时","_").replace("分","_").replace("秒","_"))
        return str.replace("年", "_").replace("月", "_").replace("日", "").replace("时", "_").replace("分", "_").replace(
            "秒", "_")


while True:
    ret,frame = vc.read()
    image = frame
    h = image.shape[0]
    w = image.shape[1]
    # 获得所有层名称与索引
    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)
    #print(lastLayer.type)
    # 检测
    blobImage = cv.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False)
    net.setInput(blobImage)
    cvOut = net.forward()

    cv.line(frame, (0, 700), (1920, 700), (0, 255, 0), 3)
    cv.line(frame, (0, 400), (1920, 400), (0, 255, 0), 3)
    cv.line(frame, (1000, 0), (1000, 1080), (0, 255, 0), 1)
    cv.putText(frame, "Up:" + str(count_up), (100, 200), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    cv.putText(frame, "Down:" + str(count_down), (100, 300), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        objIndex = int(detection[1])
        #if score > 0.1 and (objName[objIndex] =='car' or objName[objIndex] =='bus'):
        if score > 0.1 and objName[objIndex] != 'person':
            left = detection[3]*w
            top = detection[4]*h
            right = detection[5]*w
            bottom = detection[6]*h
            #print(left,right,top,bottom)
        # 绘制
            if 300<int((top+bottom)/2) < 800:
                width = int(right - left)
                height = int(bottom - top)
                cx = int((left+right)/2)
                cy = int((top+bottom)/2)
                new = True
                for i in cars:
                    if abs(cx - i.c_x) < width and abs(cy - i.c_y) < height: # 找到这辆车与上一帧中最近的车
                        new = False
                        #最开始的想法，Y轴坐标比上一帧大方向向下，比上一帧小方向向上
                        # if cy > i.c_y:
                        #     i.derection = 'down'
                        # if cy < i.c_y:
                        #     i.derection = 'up'

                        i.updateCoords(cx, cy)
                        if i.c_y >= 750 and i.derection == 'down' and i.c_count == False:

                            image = frame[int(top): int(bottom), int(left):int(right)]
                            #cv.imshow("image",image)
                            save_time = time.strftime('%H{h}%M{f}%S{s}%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日',                                                                    h='时', f='分', s='秒')
                            filepath = os.path.join(CURPATH, 'archives\{}.jpg'.format(transfer_time_format(save_time)))
                            print(filepath)
                            flag_ = cv.imwrite(filepath, image)  # save vehicle image by name of time
                            if not flag_:
                                print('图片保存失败')

                            count_down += 1
                            i.c_count = True

                        if i.c_y < 350 and i.derection == 'up' and i.c_count == False:

                            image = frame[int(top): int(bottom), int(left):int(right)]
                            # cv.imshow("image",image)
                            save_time = time.strftime('%H{h}%M{f}%S{s}%Y{y}%m{m}%d{d}').format(y='年', m='月', d='日',
                                                                                               h='时', f='分', s='秒')
                            filepath = os.path.join(CURPATH, 'archives\{}.jpg'.format(transfer_time_format(save_time)))
                            print(filepath)
                            flag_=cv.imwrite(filepath,image )#save vehicle image by name of time
                            if not flag_:
                                print('图片保存失败')

                            count_up += 1
                            i.c_count = True
                    if i.c_y >= 790 or i.c_y <= 310:
                        cars.remove(i)



                if new == True:
                    p = Car(pid, cx, cy, 'unknow', False)
                    #判断方向，这里我是在没找到好的办法，只能从简，大家测试的时候需要修改
                    #如果有好的办法请您一指点
                    if p.c_x > 1000:
                        p.derection = 'up'
                    else:
                        p.derection = 'down'
                    cars.append(p)
                    pid += 1
                #print(len(cars))
                cv.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                cv.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)

                cv.putText(image, "score:%.2f, %s"%(score, objName[objIndex]),
                    (int(left) - 10, int(top) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8);

#  显示
    cv.imshow('mobilenet-ssd-demo', frame)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
# cv.imwrite("D:/Pedestrian.png", image)
cv.waitKey(0)
cv.destroyAllWindows()