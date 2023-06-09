import importlib
import os

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        task_name = file[:file.find('.py')]
        importlib.import_module('examples.srd_beam_policy.models.' + task_name)