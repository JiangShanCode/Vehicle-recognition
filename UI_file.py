#_*_ coding:gbk _*_
#coding=utf-8


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class Ui_Dialog(QMainWindow):
    def setupUi(self, Dialog):#Dialog is an instance objec of class QtWidgets.QWidget
        self.desktop = QApplication.desktop()
        #get monitor resolution
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()

        global_widget=QWidget(self)
        global_layout = QHBoxLayout(global_widget)

        Dialog.setObjectName("Dialog")
        Dialog.resize(1500, 900)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)

        # self.label = QtWidgets.QLabel(Dialog)
        # self.label.setText("实时车流量:")
        #self.label1.setFont(QFont("Microsoft YaHei", 10, 75))

        self.label_up = QtWidgets.QLabel()
        self.label_up.setText("去向车流量(辆):")
        self.label_up.setStyleSheet('''QLabel{font : 14px;font-family:Roman times;}''')
        self.line_up = QLineEdit()
        self.line_up.setReadOnly(True)
        self.line_up.setFixedWidth(50)
        self.line_up.setStyleSheet(''' QLineEdit{text-align : center;background-color : white;font: bold;border-width: 2px;border-radius: 10px;padding: 6px;height : 14px;border-style: outset;font : 14px;font-family:Roman times;}''')

        #self.line_up.setText("0")
        # self.line_up.setGeometry(QtCore.QRect(2,5,2,5))
        #self.text_up = QtWidgets.QLabel(Dialog)
        #self.text_up.setFont(QFontDialog("Microsoft YaHei", 10, 75))

        self.label_down = QtWidgets.QLabel(Dialog)
        self.label_down.setText("来向车流量(辆):")
        self.label_down.setStyleSheet('''QLabel{font : 14px;font-family:Roman times;}''')
        self.line_down = QLineEdit()
        self.line_down.setReadOnly(True)
        self.line_down.setFixedWidth(50)
        self.line_down.setStyleSheet(''' QLineEdit{text-align : center;background-color : white;font: bold;border-width: 2px;border-radius: 10px;padding: 6px;height : 14px;border-style: outset;font : 14px;font-family:Roman times;}''')
        #self.line_up.setText("0")
        #self.line_down.resize(2, 5)

        self.pushButton = QtWidgets.QPushButton(Dialog)

        self.pushButton.setStyleSheet(
            ''' QPushButton{text-align : center;background-color : white;font: bold;border-color: gray;border-width: 2px;border-radius: 10px;padding: 6px;height : 14px;border-style: outset;font : 14px;font-family:Roman times;}''')
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setStyleSheet(
            ''' QPushButton{text-align : center;background-color : white;font: bold;border-color: gray;border-width: 2px;border-radius: 10px;padding: 6px;height : 14px;border-style: outset;font : 14px;font-family:Roman times;}''')
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setStyleSheet(
            ''' QPushButton{text-align : center;background-color : white;font: bold;border-color: gray;border-width: 2px;border-radius: 10px;padding: 6px;height : 14px;border-style: outset;font : 14px;font-family:Roman times;}''')
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setStyleSheet(
            ''' QPushButton{text-align : center;background-color : white;font: bold;border-color: gray;border-width: 2px;border-radius: 10px;padding: 6px;height : 14px;border-style: outset;font : 14px;font-family:Roman times;}''')

        self.pushButton.setObjectName("pushButton")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4.setObjectName("pushButton_4")
###########################
        #todo:above is roll down automatically
        self.spacerItem = QSpacerItem(2, 5,QSizePolicy.Expanding,QSizePolicy.Expanding)
        self.spacerItem.invalidate()


        self.label = QtWidgets.QHBoxLayout()
        #self.label_space.setGeometry(2,3,2,3)

        self.label.addItem(self.spacerItem)
        self.label.addWidget(self.label_up)
        self.label.addWidget(self.line_up)
        self.label.addItem(self.spacerItem)


        self.label.addItem(self.spacerItem)
        self.label.addWidget(self.label_down)
        self.label.addWidget(self.line_down)
        self.label.addItem(self.spacerItem)

        self.label.setSpacing(0)
        self.label.setContentsMargins(0, 0, 0, 0);

        self.button_space= QtWidgets.QHBoxLayout()
        #self.button_space2= QtWidgets.QHBoxLayout()

        self.button_space.addItem(self.spacerItem)
        self.button_space.addWidget(self.pushButton)
        self.button_space.addItem(self.spacerItem)
        self.button_space.addWidget(self.pushButton_2)
        self.button_space.addItem(self.spacerItem)
        
        #self.button_space2.addItem(self.spacerItem)
        self.button_space.addWidget(self.pushButton_3)
        self.button_space.addItem(self.spacerItem)
        self.button_space.addWidget(self.pushButton_4)
        self.button_space.addItem(self.spacerItem)
   
   

        self.layout = QVBoxLayout()



        self.layout.addLayout(self.label)
        self.layout.addLayout(self.button_space)
            
        #########################################
        self.graphicsView = QtWidgets.QLabel(Dialog)
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView.setTextFormat(QtCore.Qt.RichText)#added
        self.graphicsView.setFixedWidth(self.width*0.5)
        self.graphicsView.setFixedHeight(self.height*0.6)
        self.graphicsView.setFrameShape(QtWidgets.QFrame.Box)

        #add buttons_widget and Qgraphics to vertical layout

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.graphicsView)
        left_layout.addLayout(self.layout)

        self.graphicsView_frame = QtWidgets.QLabel(Dialog)
        self.graphicsView_frame.setObjectName("graphicsView_frame")
        self.graphicsView_frame.setTextFormat(QtCore.Qt.RichText)#added
        self.graphicsView_frame.setFixedHeight(self.height/3)
        self.graphicsView_frame.setFixedWidth(self.width/3)
        self.graphicsView_frame.setFrameShape(QtWidgets.QFrame.Box)
        #table configuration:
        #https://blog.csdn.net/zhulove86/article/details/52599738/
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(0)
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)#can only choose one line
        self.tableWidget.setEditTriggers(QTableView.NoEditTriggers)
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableWidget.doubleClicked.connect(self.display_table)
        #self.tableWidget.item.setTextAlignment(Qt.AlignHCenter)
        self.tableWidget.setHorizontalHeaderLabels(['时间','种类','置信度','车牌','车牌置信度'])

        #self.tableWidget.verticalScrollBar().valueChanged.connect(lambda :print(1))
        
        ##################
        VerticalLayout = QtWidgets.QVBoxLayout()
        #VerticalLayout.addWidget(self.label)
        VerticalLayout.addWidget(self.tableWidget)
        VerticalLayout.addWidget(self.graphicsView_frame)


        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addLayout(VerticalLayout)

        #######################################
        menubar=self.menuBar()
        menubar.setNativeMenuBar(False)
        export=QAction('导出成txt',self)
        export_txt = QAction('导出成txt',self)
        set_cam=QAction('设置视频流',self)
        fileMenu =menubar.addMenu('文件')
        export_menu = fileMenu.addMenu('导出')
        export_menu.addAction(export)
        export_menu.addAction(export_txt)
        setting = menubar.addMenu(' 设置 ')
        #about = menubar.addMenu(' 关于 ')
        #about.addAction('关于')
        set_rec=QAction('设置矩形框比例',self)
        show_rec=QAction('显示/隐藏矩形框',self)
        set_interv = QAction('设置截帧间隔',self)
        set_iou = QAction('设置IOU',self)
        setting.addAction(set_interv)
        setting.addAction(set_rec)
        setting.addAction(show_rec)
        setting.addAction(set_cam)
        setting.addAction(set_iou)
        #############################
        #about.triggered[QAction].connect(lambda:QMessageBox.about(self,'about','constructed by pt'))
        set_rec.triggered.connect(self.set_rectangle)
        show_rec.triggered.connect(self.show_rectangle)
        set_interv.triggered.connect(self.set_interv)
        export.triggered.connect(Dialog.export)
        export_txt.triggered.connect(Dialog.export_txt)
        set_cam.triggered.connect(self.set_camera)
        set_iou.triggered.connect(self.set_IOU)
        #################
        global_layout.addLayout(left_layout)
        global_layout.addLayout(right_layout)
        self.retranslateUi(Dialog)
        self.pushButton_2.clicked.connect(Dialog.openimage)
        self.pushButton.clicked.connect(Dialog.inquiry)#inquiry
        self.pushButton_3.clicked.connect(Dialog.webcamera)#inquiry
        self.pushButton_4.clicked.connect(Dialog.export)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        #self.setLayout(global_layout)
        self.setCentralWidget(global_widget)
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "车辆识别系统"))
        self.pushButton.setText(_translate("Dialog", "   "))
        self.pushButton_2.setText(_translate("Dialog", "导入视频"))
        self.pushButton_3.setText(_translate("Dialog", "启动摄像"))
        self.pushButton_4.setText(_translate("Dialog", "导出数据"))

    def set_rectangle(self):
        value1_, ok_1 =QInputDialog.getDouble(QWidget(),'get para','输入xmin占视频宽度比例',0.3,0,1,3)
        value2_, ok_2 =QInputDialog.getDouble(QWidget(),'get para','输入xmax占视频宽度比例',0.6,0,1,3)
        value3_, ok_3 =QInputDialog.getDouble(QWidget(),'get para','输入ymin占视频高度比例',0.3,0,1,3)
        value4_, ok_4 =QInputDialog.getDouble(QWidget(),'get para','输入ymax占视频高度比例',0.6,0,1,3)
        if ok_1 and ok_2 and ok_3 and ok_4:
            try:
                self.yolo_thread.BOUNDING=[value1_ ,value2_ ,value3_ ,value4_]
            except Exception as e:
                QMessageBox.warning(self,'error ',repr(e))
    def show_rectangle(self):
        if self.VIS_RECGTANGLE:
            self.VIS_RECGTANGLE=0
        else:
            self.VIS_RECGTANGLE=1

    def set_interv(self):
        value_, ok_ =QInputDialog.getDouble(QWidget(),'get para',' 输入截帧间隔时间',0.5,0,10,2)
        if ok_:
            self.INTERVAL = value_
            
    def set_camera(self):
        #value_, ok_ =QInputDialog.getInt(QWidget(),'get para',' input webcamera ID!',0,0,500)
        value_, ok_ =QInputDialog.getText(QWidget(),'get para',' input web ID')
        if ok_:
            self.CAMID=value_
            
    def set_IOU(self):
        value1_, ok_1 =QInputDialog.getDouble(QWidget(),'get para','输入IOU',0.0,0,1,3)
        if ok_1:
            self.IOU=value1_








