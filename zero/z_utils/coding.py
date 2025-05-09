
import os
import re
from numpy import array as npa


def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
