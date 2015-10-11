import config_biopax_model as cbm


if cbm.is_in_use:
    import parse_model_frm_csv_biopax as pmcb
    bm_obj = pmcb.BioPAXModel()

