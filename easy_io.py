import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    input = raw_input
except NameError:
    input = input


def confirm(prompt, yes='y', no='n'):
    t = None
    while t not in [yes, no]:
        t = input(prompt)
    if t == yes:
        return True
    elif t == no:
        return False
    else:
        raise ValueError


def confirm_file_overwrite(file_path):
    if os.path.isfile(file_path):
        if not confirm('"{}" already exists, overwrite? (y/n): '.format(file_path)):
            return False
    return True


def read_pkl_file(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        return pickle.load(f)


def write_pkl_file(pkl_file_path, variable):
    if not os.path.isdir(os.path.dirname(pkl_file_path)):
        os.makedirs(os.path.dirname(pkl_file_path))
    if confirm_file_overwrite(pkl_file_path):
        with open(pkl_file_path, 'wb') as g:
            pickle.dump(variable, g, 0)


def make_dir(*args):
    for dir_path in args:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        else:
            print '%s exists.' % dir_path
