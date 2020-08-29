import sys
from functools import partial
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

from PyQt5.QtWidgets import (QApplication, QMainWindow, QMenu, QVBoxLayout, 
                             QSizePolicy, QMessageBox, QWidget, QPushButton, 
                             QAction, QHBoxLayout, QLabel, QLineEdit, QSpacerItem)
from PyQt5.QtGui import QIcon

from PyQt5.QtCore import QSize

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import random

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class WidgetPlot(QWidget):
    def __init__(self, parent=None, width=10, height=4, *args, **kwargs):
        QWidget.__init__(self, parent=parent, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = PlotCanvas(self, width=width, height=height)
        #self.toolbar = NavigationToolbar(self.canvas, self)
        #self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

    def plot(self, data, *args, **kwargs):
        self.canvas.plot(data=data, *args, **kwargs)

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.tight_layout()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Prediction Probabilities')
        self.ax.set_xlabel("Class")
        self.ax.set_xticks([])

    def plot(self, data, clear=False):
        if clear:
            del self.ax.lines[:]
            del self.ax.patches[:]

        cmap = matplotlib.cm.get_cmap("RdYlGn")
        self.ax.bar(np.arange(data.shape[-1]), data, color=cmap(data), edgecolor="k")
        self.ax.relim()
        self.ax.set_xticks(np.arange(data.shape[-1]))
        self.draw()


class ButtonMenu(QWidget):
    def __init__(self, parent=None, existing_classes=1, predict_btn_callback=None, class_btn_callback_func=None, add_class_func=None, *args, **kwargs):
        super(ButtonMenu, self).__init__(parent=parent, *args, **kwargs)
        self.class_buttons = []
        self.button_height = 25
        self.button_padding = 5
        self.setWindowTitle("PyQt5 Button Click Example")

        self.setLayout(QVBoxLayout())
        self.layout().setSpacing(1)
        self.layout().setAlignment(QtCore.Qt.AlignTop)
        # prediction button
        self.predict_button = QPushButton("Predict", self)
        if predict_btn_callback is not None:
            self.predict_button.clicked.connect(predict_btn_callback)
        self.layout().addWidget(self.predict_button)

        # new class button
        self.new_class_button = QPushButton("New Class", self)
        self.new_class_button.clicked.connect(self._new_button)
        self.layout().addWidget(self.new_class_button)

        if add_class_func is None:
            add_class_func = lambda *args, **kwargs: None
        self._add_class_func = add_class_func

        self.class_btn_callback_func = class_btn_callback_func

        # add existing class buttons
        for c in range(existing_classes):
            self._new_button(is_clicked=False)


    def _new_button(self, is_clicked=True):
        if is_clicked:
            self._add_class_func()

        num_buttons = len(self.class_buttons)
        newBtn = QPushButton(f'Class {num_buttons}', self)
        newBtn.show()

        def _button_clicked():
            print(f"Button {num_buttons} clicked")
            if self.class_btn_callback_func is not None:
                self.class_btn_callback_func(num_buttons)
        
        newBtn.clicked.connect(_button_clicked)
        self.class_buttons.append(newBtn)
        self.layout().addWidget(newBtn)

class Window(QtWidgets.QMainWindow):
    def __init__(self, predict_btn_callback=None, 
                 class_btn_callback_func=None, add_class_func=None, *args, **kwargs):
        super(QtWidgets.QMainWindow, self).__init__(*args, **kwargs)
        self.button_menu = ButtonMenu(parent=self, 
                                      predict_btn_callback=predict_btn_callback,
                                      class_btn_callback_func=class_btn_callback_func,
                                      add_class_func=add_class_func)


        # Create the maptlotlib FigureCanvas object, 
        # which defines a single set of axes as self.axes.
        #self.plot_widget = PlotCanvas(self, width=5, height=4)
        #self.plot_widget = WidgetPlot(self)
        #self.setCentralWidget(self.plot_widget)

        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addWidget(self.button_menu)
        #self.hbox.addWidget(self.plot_widget)

        
        self.setLayout(self.hbox)

        self.button_menu.show()
        self.show()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, 
                 existing_classes=1, 
                 predict_btn_callback=None, 
                 class_btn_callback_func=None,
                 add_class_func=None,
                 *args,
                 **kwargs):
        super(QtWidgets.QMainWindow, self).__init__(*args, **kwargs)

        self.title = 'test'
        self.left = 20
        self.top = 50
        self.width = 640
        self.height = 360

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        container_widget = QWidget(self)
        self.setCentralWidget(container_widget)


        self.button_menu = ButtonMenu(parent=container_widget, 
                                      existing_classes=existing_classes,
                                      predict_btn_callback=predict_btn_callback,
                                      class_btn_callback_func=class_btn_callback_func,
                                      add_class_func=add_class_func)

        self.plot_widget = WidgetPlot(parent=container_widget)

        
        hlay = QHBoxLayout(container_widget)
        hlay.addWidget(self.button_menu)
        hlay.addWidget(self.plot_widget)

    def plot(self, data, *args, **kwargs):
        self.plot_widget.plot(data, *args, **kwargs)


    def _setup_menu(self):
        self.statusBar().showMessage('Ready')

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        fileMenu = mainMenu.addMenu('File')
        helpMenu = mainMenu.addMenu('Help')

        exitButton = QAction(QIcon('exit24.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

    def clickMethod(self):
        print('Clicked Pyqt button.')
        if self.line.text() == '':
            self.statusBar().showMessage('Not a Number')
        else:
            print('Number: {}'.format(float(self.line.text())*2))
            self.statusBar().showMessage('Introduction of a number')
            self.nameLabel2.setText(str(float(self.line.text())*2))


class Application(QApplication):
    def __init__(self, *args, **kwargs):
        super(QApplication, self).__init__(*args, **kwargs)

if __name__ == '__main__':
    app = Application(sys.argv)
    w = MainWindow()
    
    data = np.random.normal(size=(6))
    data /= data.sum()
    
    w.plot(data)

    w.show()
    sys.exit(app.exec_())
