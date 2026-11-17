import argparse
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')

def run(script_or_args):
    env = os.environ.copy()
    env['PYTHONPATH'] = SRC_DIR + os.pathsep + env.get('PYTHONPATH','')
    cmd = [sys.executable] + script_or_args
    print('>', ' '.join(cmd))
    subprocess.run(cmd, cwd=BASE_DIR, env=env, check=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit',action='store_true')
    p.add_argument('--adapters',action='store_true')
    p.add_argument('--preprocess',action='store_true')
    p.add_argument('--tests',action='store_true')
    p.add_argument('--core',action='store_true',help='Run verified core benchmark from raw midterm CSVs')
    p.add_argument('--evaluate',action='store_true',help='Aggregate verified results; never fabricates temporal/OOD metrics')
    p.add_argument('--report',action='store_true')
    p.add_argument('--all',action='store_true')
    a=p.parse_args()

    if a.all or a.audit: run(['src/preprocessing/audit.py'])
    if a.all or a.adapters: run(['src/adapters/run_adapters.py'])
    if a.all or a.preprocess: run(['src/preprocessing/run_preprocessing.py'])
    if a.all or a.tests: run(['-m','pytest','-q','tests'])
    if a.all or a.core: run(['run_verified_core.py'])
    if a.all or a.evaluate: run(['src/evaluation/protocols.py'])
    if a.all or a.report: run(['src/evaluation/reporting.py'])

if __name__=='__main__': main()
