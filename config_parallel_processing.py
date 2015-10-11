import config_hpcc as ch
from config_console_output import *


if ch.is_hpcc and ch.is_hpcc_machine:
    num_cores = 7
else:
    num_cores = 3





