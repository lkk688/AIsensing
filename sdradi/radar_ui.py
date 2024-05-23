
Runtime="QT6"
if Runtime=="Side6": #"QT5" Side6 and QT5 works in Mac
   from PyQt5.QtWidgets import QApplication, QWidget, QLabel
   #from PyQt5.QtGui import QIcon
   #from PyQt5.QtCore import pyqtSlot
elif Runtime=="QT5":
   #from PyQt5.QtWidgets import QApplication, QWidget, QLabel
   from PyQt5.QtCore import Qt
   from PyQt5.QtWidgets import *
elif Runtime=="QT6":
   #from PyQt6.QtWidgets import QApplication, QWidget, QLabel
   from PyQt6.QtCore import Qt
   from PyQt6.QtWidgets import *
elif Runtime=="Side6":
   from PySide6 import QtCore, QtGui, QtWidgets
   from PySide6.QtWidgets import QApplication, QWidget, QLabel

#from PyQt5.QtCore import Qt
#from PyQt5.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
