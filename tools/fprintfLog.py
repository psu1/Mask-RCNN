'''
Save logs to txt file.
'''
import os
import yaml

def remove(file_name):
    try:
        os.remove(file_name)
    except:
        pass

def fprintf_log(msg, file=None, init=False, additional_file=None):

    _file = file + '/logs.txt'

    if file is None:
        pass
    else:
        if init:
            remove(file)
        with open(_file, 'a') as log_file:
            log_file.write('%s\n' % msg)

        if additional_file is not None:
            # TODO (low): a little buggy here: no removal of previous additional_file
            with open(additional_file, 'a') as addition_log:
                addition_log.write('%s\n' % msg)


def  write_to_yaml(msg, file=None):

    _file = file + 'cfg.yaml'

    with open(_file, 'w') as outfile:
        yaml.dump(msg, outfile, default_flow_style=False)

