import re


alphanum_regex = re.compile('[^a-zA-Z0-9]')
alpha_regex = re.compile('[^a-zA-Z]')
no_quotes_regexp = re.compile('^"|"|\'$')
#
concept_regexp = re.compile('[a-z]+[-][0-9][0-9]')
concept_num_regexp = re.compile('[-][0-9][0-9]')

root = 'root'
