import functools
import os
import re
from importlib import import_module
from typing import Callable, Pattern

import yaml
from livelossplot.outputs import NeptuneLogger


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


def create_experiment(func: Callable) -> Callable:
    """Create experiment with function keyword parameters and generated name"""
    def wrapper(*args, **params):
        neptune_project_name = params.get('neptune_project')
        output_dir = params['output_dir']
        del params['output_dir']
        logger_outputs = ['ExtremaPrinter']
        params['logger_outputs'] = logger_outputs
        if neptune_project_name is not None:
            del params['neptune_project']
            neptune_output = NeptuneLogger(
                project_qualified_name=neptune_project_name, params=params, upload_source_files='**/*.py'
            )
            logger_outputs.append(neptune_output)
            params['run_dir'] = os.path.join(output_dir, neptune_output.experiment.id)
            ret = func(*args, **params)
            neptune_output.neptune.stop()
        else:
            params['run_dir'] = create_artifacts_dir(output_dir)
            ret = func(*args, **params)
        return ret

    return wrapper


def import_function(class_path: str) -> Callable:
    """Function take module with to class or function and imports it dynamically"""
    modules = class_path.split('.')
    module_str = '.'.join(modules[:-1])
    cls = modules[-1]
    module = import_module(module_str)
    return getattr(module, cls)
