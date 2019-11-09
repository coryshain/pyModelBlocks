from flask import render_template

from mb.gui.app import app


@app.route('/')
@app.route('/index')
def index():

    out = render_template(
        'index.html',
        title='ModelBlocks Home'
    )

    return out