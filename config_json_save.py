import config_hpcc as ch
from config_console_output import *


if ch.is_hpcc:
    num_interactions_every_save = 20
else:
    num_interactions_every_save = 2







