import sys
Runtime="QT6"#"QT6" works in windows
if Runtime=="Side6": #"QT5" Side6 and QT5 works in Mac
   from PyQt5.QtWidgets import QApplication, QWidget, QLabel
   #from PyQt5.QtGui import QIcon
   #from PyQt5.QtCore import pyqtSlot
elif Runtime=="QT5":
   from PyQt5.QtWidgets import QApplication, QWidget, QLabel
elif Runtime=="QT6":
   from PyQt6.QtWidgets import QApplication, QWidget, QLabel
elif Runtime=="Side6":
   from PySide6 import QtCore, QtGui, QtWidgets
   from PySide6.QtWidgets import QApplication, QWidget, QLabel


def window():
   app = QApplication(sys.argv)
   widget = QWidget()

   textLabel = QLabel(widget)
   textLabel.setText("Hello World!")
   textLabel.move(110,85)

   widget.setGeometry(50,50,320,200)
   widget.setWindowTitle("PyQt5/6 Example")
   widget.show()

   if Runtime=="QT5":
      sys.exit(app.exec_())
   elif Runtime in ["QT6", "Side6"]:
      app.exec()

if __name__ == '__main__':
   window()