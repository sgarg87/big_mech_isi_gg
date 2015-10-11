import re


def extract_original_amr_dot_file_name(file_name):
    num_dots = len(re.findall('\.dot', file_name))
    if num_dots < 1 or num_dots > 2:
        raise AssertionError
    elif num_dots == 2:
        if file_name.endswith('.dot'):
           file_name.strip('.dot')
        else:
            raise NotImplementedError
    org_amr_file_paths = re.findall(r'.+\.dot', file_name)
    if len(org_amr_file_paths) != 1:
        raise AssertionError
    return org_amr_file_paths[0]
