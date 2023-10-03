import numpy as np
import matplotlib.pyplot as plt

class UTILS(object):
    """
    Helper class with GENERIC FUNCTIONS
    """
    def __init__(self,log_name):
        self.log_name = log_name
        pass

    def write_log(self,message):
        # Tell the log
        fid_log = open(self.log_name,'a')
        fid_log.write(message)
        fid_log.close()

    def func_pass(self,*args,**kwargs):
        pass

