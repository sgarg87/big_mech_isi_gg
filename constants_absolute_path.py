import config_hpcc as ch
from config_console_output import *


if ch.is_hpcc and ch.is_hpcc_machine:
	absolute_path = '/auto/rcf-proj2/gv/sahilgar/big_mech_research_coding/biopathways/models/'
	# absolute_path = ''
else:
	absolute_path = ''



