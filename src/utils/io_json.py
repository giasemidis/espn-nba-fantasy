import os.path
import sys
import json


def valid_file(func):
    """
    Checks the validity of the input file of read-data fuction. If files
    does not exist, it exists.
    """

    def wrapper(filename, *args, **kwargs):
        if os.path.isfile(filename):
            a = func(filename, *args, **kwargs)
            return a
        else:
            sys.exit('File %s not found' % filename)
            return
    return wrapper


def valid_folder(func):
    """
    Checks the validity of the output directory of a write-data function. If
    directory does not exist, it exists.
    """

    def wrapper(filepath, *args, **kwargs):
        dirname = os.path.dirname(filepath)
        dirname = dirname if dirname != '' else '.'
        if os.path.isdir(dirname):
            a = func(filepath, *args, **kwargs)
            return a
        else:
            sys.exit('Directory %s not found' % filepath)
            return
    return wrapper


@valid_folder
def write_json(file, data):
    '''
    Writes data into json file
    '''
    with open(file, 'w', encoding='utf8') as outfile:
        json.dump(data, outfile, ensure_ascii=False)
    return


@valid_file
def read_json(file):
    '''
    Reads data from json file
    '''
    with open(file, 'r', encoding='utf8') as outfile:
        data = json.load(outfile)
    return data
