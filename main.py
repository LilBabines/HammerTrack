import sys

from PySide6 import QtWidgets
from src.windows import LauncherWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LauncherWindow()
    w.show()
    app.exec()

if __name__ == "__main__":

    main()
