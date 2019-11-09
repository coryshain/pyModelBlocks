from flask import render_template, flash, redirect, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

from mb.gui.app import app
from mb.core.general.core import CONFIG_PATH, USER, USER_SETTINGS, DEFAULT_SETTINGS

keys = DEFAULT_SETTINGS.keys()

attr_dict = {}
for k in keys:
    new_field = StringField(k)
    attr_dict[k] = new_field
attr_dict['save'] = SubmitField('Save')
attr_dict['exit'] = SubmitField('Exit')
ConfigForm = type('ConfigForm', (FlaskForm,), attr_dict)

@app.route('/config', methods=['GET', 'POST'])
def config():
    form = ConfigForm()

    kwargs = {
        'form': form,
        'keys': keys
    }

    for k in keys:
        getattr(form, k).default = USER_SETTINGS.get(k, DEFAULT_SETTINGS.get(k, ''))

    if form.validate_on_submit():
        if form.save.data:
            print('Save was pressed')
            for k in DEFAULT_SETTINGS:
                USER_SETTINGS[k] = getattr(form, k).data.replace('%', '%%')

            with open(CONFIG_PATH, 'w') as f:
                USER.write(f)

            return redirect('/config')
        else:
            print('Exit was pressed')
            return redirect('/index')

    form.process()

    out = render_template(
        'config.html',
        title='Edit ModelBlocks Configuration',
        **kwargs
    )

    return out
#
# @app.route('/get_config_data', methods=['POST'])
# def get_config_data():
#     out = {}
#     for k in config_dict:
#         out[k] = request.form[k]
#
#     print(out)
#
#     return out