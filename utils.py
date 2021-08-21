import functools
import os
import re
from typing import Callable, Pattern

import yaml


def unpack_config(func: Callable) -> Callable:
    """Load parameters from a config file and inject it to function keyword arguments"""
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        config_file = kwargs.get('config')
        if config_file:
            del kwargs['config']
            with open(config_file, 'r') as f:
                run_args = yaml.full_load(f)
                kwargs.update(run_args)
        ret = func(*args, **kwargs)
        return ret

    return _wrapper


def create_artifacts_dir(runs_dir: str, run_template: Pattern = re.compile(r'(offline-experiment-)([0-9]+)')) -> str:
    os.makedirs(runs_dir, exist_ok=True)
    runs = [re.match(run_template, run) for run in os.listdir(runs_dir) if re.match(run_template, run)]
    if len(runs) == 0:
        next_run_dir = 'offline-experiment-0'
    else:
        last_run_match = max(runs, key=lambda r: int(r.group(2)))
        next_run_id = int(last_run_match.group(2)) + 1
        next_run_dir = last_run_match.group(1) + str(next_run_id)
    next_run_dir = os.path.join(runs_dir, next_run_dir)
    os.makedirs(next_run_dir)
    return next_run_dir
