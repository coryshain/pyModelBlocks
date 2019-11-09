import os
from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret-key'

from mb.gui.app import index, config
