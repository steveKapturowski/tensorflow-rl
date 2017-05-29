#!/usr/bin/env python
import subprocess
import argparse
import signal
import yaml


def launch_proc(proc_type, num_workers, task_index, args):
	cmd = ['python', 'main.py'] + args \
		+ ['--job_name', proc_type, '--task_index', str(task_index), '-n', str(num_workers)]
	return subprocess.Popen(cmd)


def launch_cluster(spec, arg_string, daemonize=False):
	num_workers = len(spec['worker'])
	parameter_servers = [launch_proc('ps', num_workers, i, arg_string)
		for i, _ in enumerate(spec['ps'])]
	workers = [launch_proc('worker', num_workers, i, arg_string)
		for i, _ in enumerate(spec['worker'])]

	if not daemonize:
		procs = parameter_servers + workers
		try:
			for p in procs:
				p.wait()
		except KeyboardInterrupt:
			for p in procs:
				# p.send_signal(signal.SIGINT)
				p.kill()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--num_workers', default=8, type=int, help='number of workers to use if in local mode', dest='num_workers')
	parser.add_argument('--cluster_config', default=None, help='load cluster config from yaml file', dest='cluster_config')
	parser.add_argument('--daemonize', action='store_true', help='detach after cluster creation', dest='daemonize')
	args, unused_args = parser.parse_known_args()

	if args.cluster_config:
		with open(args.cluster_config, 'r') as f:
			spec = yaml.load(f)
	else:
		spec = {
			'ps': ['localhost:2048'],
			'worker': [
				'localhost:{}'.format(4096+i)
				for i in range(args.num_workers)
			]}

	launch_cluster(spec, unused_args, daemonize=args.daemonize)


