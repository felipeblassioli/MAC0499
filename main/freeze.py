from flask.ext.frozen import Freezer
from senjo.web import create_app

app = create_app()
app.config['FREEZER_DESTINATION'] = 'frozen-build'
app.config['FREEZER_RELATIVE_URLS'] = True
freezer = Freezer(app)

if __name__ == '__main__':
    print 'freezing begin'
    freezer.freeze()
    print 'freezing done'
