from setuptools import setup, find_packages

setup(
	name='Senjo',
	version='0.0.1',
	description='',
	packages=find_packages(),
	include_package_data=True,
	zip_safe=False,
	author='Felipe Blassioli',
	author_email='felipeblassioli@gmail.com',
	install_requires=[
		'Flask>=0.8',
		'peewee>=2.6.2',
		'flask-peewee>=0.6.4',
		'PyMySQL',
		'flask-admin',
		'six'
	]
)
