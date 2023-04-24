import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import sys

class MyButton(qtw.QPushButton):
    """Subclassed QPushButton, with the follwing parameters:
        name -- string with the name of the button
        functions -- if == 0: waarning message
                     if list of functions: called (in order) by the click
        tooltip -- hint for when mouse hovers on the button
        size -- button dimension (default [110, 30])
        style -- name of the stylesheet
    """
    def __init__(self,
                 name='button name',
                 functions=0,
                 tooltip='button description',
                 size=[90, 30],
                 style='fancy'):
        super().__init__(name)
        self.setToolTip(tooltip)
        self.size = qtc.QSize(size[0], size[1])
        self.setFixedSize(self.size)
        self.setObjectName(style)
        if(functions==0):
            print('\nbutton '+str(name)+' connected to NO function.\n')
        else:
            for i in range (len(functions)):
                self.clicked.connect(functions[i])

class Choice(qtw.QComboBox):
    """Subclassed QComboBox, with the follwing parameters:
        items -- list of choices (strings)
        functions -- if == 0: warning message
                     if lenght == 1: function called on activation
                     if lzngth == # of items: item-function ordered calls
                     else: error message
        tooltip -- hint for when mouse hovers on the box
        alignment -- default is AlignCenter
        style -- name of the stylesheet, default is "fancy"
    """
    def __init__(self,
                 items=0,
                 functions=0,
                 tooltip='description',
                 style='fancy'):
        super().__init__()
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setAlignment(qtc.Qt.AlignCenter)
        self.setFixedSize(40, 20)
        self.setObjectName(style)
        if items != 0:
            self.addItems(items)
            self.items = items
        if functions == 0 : 
            print('\ncombobox connected to NO function.\n')
        elif len(functions) == 1: 
            self.activated.connect(functions[0])
        elif len(functions) == len(items):
            self.functions = functions
            self.activated.connect(self.chosen_function)
        else: print('\nNumebr of functions and choices mismatch.\n')
    def chosen_function(self):
        self.functions[self.currentIndex()]()

class InputWidget(qtw.QWidget):
    """Subclassed Label + Inpu, with the follwing parameters:
        label -- name of the label
        default_input -- 0 is the default value inside
        functions -- if == 0: warning message
                     else: add functions list to be called IN ORDER
        tooltip -- hint for when mouse hovers on the box
        alignment -- default is AlignCenter
        style -- name of the stylesheet, default is "fancy"
    """
    def __init__(self,
                 label='label',
                 default_input='0',
                 functions=0,
                 tooltip='description',
                 alignment=qtc.Qt.AlignCenter,
                 style='fancy', 
                 readonly = False):
        super().__init__()
        self.label = qtw.QLabel(label)
        self.label.setToolTip(tooltip)
        self.label.setAlignment(alignment)
        self.label.setObjectName("fancy")
        self.input = qtw.QLineEdit(default_input)
        self.input.setFixedSize(50, 20)
        self.input.setAlignment(alignment)
        self.input.setObjectName(style)
        if(readonly == True): self.input.setReadOnly(True)
        self.layout = qtw.QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.input)
        self.setLayout(self.layout)
        if(functions==0):
            print('\nInput '+str(label)+' connected to NO function.\n')
        else:
            for i in range (len(functions)):
                self.input.textChanged.connect(functions[i])