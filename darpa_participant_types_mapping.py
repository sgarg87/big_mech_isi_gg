import darpa_participant_types as dpt
from config_console_output import *


def get_darpa_protein_type(org_type):
    if org_type is None or not org_type.strip():
        return org_type
    org_type = org_type.lower()
    if org_type in dpt.list:
        return org_type
    new_type = None
    if 'complex' in org_type:
        new_type = dpt.complex
    elif 'gene' in org_type:
        new_type = dpt.gene
    elif 'enzy' in org_type:
        new_type = dpt.protein
    elif 'famil' in org_type:
        new_type = dpt.protein_family
    elif 'molecule' in org_type:
        new_type = dpt.chemical
    elif 'chemi' in org_type:
        new_type = dpt.chemical
    return new_type
