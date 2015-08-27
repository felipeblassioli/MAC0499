from fabric.api import *

env['hosts'] = [ 'fblassioli@linux.ime.usp.br' ]
pack_dir = 'senjo/web/frozen-build'
packed_file = 'site.zip'
target_dir = 'www/mac0499'

def pack():
	with prefix('. env/bin/activate'):
		local('python freeze.py')

	local('mv %s/index.html %s/demo.html' % (pack_dir,pack_dir))
	local('cp ../site/index.html %s/' % pack_dir)
	with lcd(pack_dir):
		local('zip -r %s .' % packed_file)
	local('mv %s/%s .' % (pack_dir, packed_file))

def deploy():
	put(packed_file, target_dir)
	with cd(target_dir):
		run('unzip %s' % packed_file)
