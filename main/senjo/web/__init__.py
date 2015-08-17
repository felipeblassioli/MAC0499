from flask import Flask, render_template, send_from_directory, send_file, jsonify

def create_app(package_name=__name__, settings_override=None):
	app = Flask(package_name, instance_relative_config=True)
	app.config.from_object('senjo.web.config.DefaultConfiguration')
	app.config.from_pyfile('senjo.cfg', silent=True)
	app.config.from_envvar('SENJO_SETTINGS', silent=True)
	if type(settings_override) == dict:
		app.config.update(settings_override)
	else:
		app.config.from_object(settings_override)

	@app.route('/')
	def index():
		import os
		import json
		print app.instance_path
		experiments_dir = os.path.join(app.instance_path, app.config['EXPERIMENTS_DIR'])
		experiments = []
		for f in os.listdir(experiments_dir):
			if f.endswith('json'):
				with open(os.path.join(experiments_dir, f)) as fp:
					exp = json.load(fp)
					exp['name'] = f
					experiments.append(exp)
		datasets = list(set([ exp['training_dataset']['name'] for exp in experiments ]))

		return render_template('index.html', experiments=experiments, datasets=datasets)

	@app.route('/download/<path:filename>')
	def download_image(filename):
		import os
		data_dir = app.config['DATASETS_DIR']
		filename = filename.replace('instance/'+data_dir+'/', '')
		if not data_dir.startswith('/'):
			data_dir = os.path.join(app.instance_path, data_dir)
		full_path = os.path.join(data_dir, filename)
		return send_file(full_path)
	return app
