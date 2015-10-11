import sys
import os
import config_hpcc as ch
import io

other_io = open(os.devnull, 'w')


if ch.is_hpcc:
    sys.stdout = other_io
