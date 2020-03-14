from classification_tools import *
from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog, QLineEdit, QAction, QFrame, QHBoxLayout, QLabel, QMainWindow, QApplication, QPushButton, QPlainTextEdit, QMessageBox
from PyQt5.QtCore import pyqtSlot, Qt

NORMAL_LABEL_STYLESHEET = "color: rgb(43, 62, 81); background-color: rgb(228, 231, 234); font-size: 16px; font-family: Arial; border-width: 3px; border-style: solid; border-color: rgb(228, 231, 234); border-radius: 20px; width: 20px; height: 8px;"
HIGHLIGHT_LABEL_STYLESHEET = "color: rgb(43, 62, 81); background-color: rgb(228, 231, 234); font-size: 16px; font-family: Arial; border-width: 3px; border-style: solid; border-color: rgb(0, 240, 0); border-radius: 20px; width: 20px; height: 8px;"

Classificator.init_stopwords_default() # Initializing stopwords class
vocabularies = list()
labels = list()

class Ui(QMainWindow):
    def __init__(self, ui_filename):
        super(Ui, self).__init__()
        uic.loadUi(ui_filename, self)
        self.setFixedSize(self.size())
        self.classify = self.findChild(QPushButton, 'Classify')
        self.classify.clicked.connect(self.clicked_classify)
        self.textArea = self.findChild(QPlainTextEdit, 'TextArea')
        self.textAreaLabel = self.findChild(QLabel, 'TextAreaLabel')
        self.container = self.findChild(QHBoxLayout, 'Container')
        self.classifyFrame = self.findChild(QFrame, 'ClassifyFrame')
        self.modelLabel = self.findChild(QLabel, 'Label')
        self.actionClassify = self.findChild(QAction, 'actionStart_Classification')
        self.actionClassify.triggered.connect(self.triggered_action_classify)
        self.actionAddModel = self.findChild(QAction, 'actionAdd')
        self.actionAddModel.triggered.connect(self.triggered_action_add_model)
        self.addFrame = self.findChild(QFrame, 'AddModelFrame')
        self.browse = self.findChild(QPushButton, 'Browse')
        self.browse.clicked.connect(self.clicked_browse)
        self.fileNameLabel = self.findChild(QLabel, 'FileNameLabel')
        self.load = self.findChild(QPushButton, 'Load')
        self.load.clicked.connect(self.clicked_load)
        self.selectLabel = self.findChild(QLabel, 'SelectLabel')
        self.fileName = self.findChild(QLineEdit, 'FileName')
        self.addFrame.setVisible(False)
        self.modelLabel.setVisible(False)
        self.show()

    @pyqtSlot()
    def clicked_classify(self):
        if not self.textArea.toPlainText():
            self.show_alert("Void Text Error", "Text Area cannot be void")
        elif not vocabularies:
            self.show_alert("No Vocabulary", "No Vocabulary loaded.\nGo to \"Models->Add\" to add a model")
        else:
            labelled_text = LabelledText(self.textArea.toPlainText(), vocabularies, cleaning_level=HIGH, fast=True)
            for l in labels:
                if l.text() == labelled_text.get_label():
                    l.setStyleSheet(HIGHLIGHT_LABEL_STYLESHEET)
                else:
                    l.setStyleSheet(NORMAL_LABEL_STYLESHEET)
    @pyqtSlot()
    def clicked_browse(self):
        self.fileName.setText(QFileDialog.getOpenFileName()[0])

    @pyqtSlot()
    def clicked_load(self):
        v = Vocabulary.load(self.fileName.text())
        if not v:  # if v == None
            self.show_alert("Error", "Error while loading model")
        elif v in vocabularies:
            self.show_alert("Warning", "Model already loaded")
        else:
            vocabularies.append(v)
            l = self.new_model_label(v.label)
            labels.append(l)
            self.container.addWidget(l)

    def triggered_action_classify(self):
        self.classifyFrame.setVisible(True)
        self.addFrame.setVisible(False)

    def triggered_action_add_model(self):
        self.addFrame.setVisible(True)
        self.classifyFrame.setVisible(False)

    def show_alert(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def new_model_label(self, text):
        l = QLabel()
        l.setText(text)
        l.setAlignment(Qt.AlignCenter)
        l.setMinimumHeight(41)
        l.setMinimumWidth(121)
        l.setStyleSheet(NORMAL_LABEL_STYLESHEET)
        return l

def main():
    app = QApplication([])
    window = Ui('gui/gui.ui')
    app.exec_()


if __name__ == "__main__":
    main()