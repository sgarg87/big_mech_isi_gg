import process_dot_and_infer as pdi


paths = [
    "../../mitre_reeval/mitre_reeval_PMC2174844",

    "../../mitre_reeval/mitre_reeval_PMC2173797",

    "../../mitre_reeval/mitre_reeval_PMC2173577",

    "../../mitre_reeval/mitre_reeval_PMC2172734",

    "../../mitre_reeval/mitre_reeval_PMC2172453",

    "../../mitre_reeval/mitre_reeval_PMC2171478",

    "../../mitre_reeval/mitre_reeval_PMC2156209",

    "../../mitre_reeval/mitre_reeval_PMC2118081",

    "../../mitre_reeval/mitre_reeval_PMC1392235",

    "../../mitre_reeval/mitre_reeval_PMC1240052",

    "../../mitre_reeval/mitre_reeval_PMC2585478",

    "../../mitre_reeval/mitre_reeval_PMC2518715",

    "../../mitre_reeval/mitre_reeval_PMC2442201",

    "../../mitre_reeval/mitre_reeval_PMC2212462",

    "../../mitre_reeval/mitre_reeval_PMC2196260",

    "../../mitre_reeval/mitre_reeval_PMC2196252",

    "../../mitre_reeval/mitre_reeval_PMC2194190",

    "../../mitre_reeval/mitre_reeval_PMC2193513",

    "../../mitre_reeval/mitre_reeval_PMC2193139",

    "../../mitre_reeval/mitre_reeval_PMC3691183",

    "../../mitre_reeval/mitre_reeval_PMC3594181",

    "../../mitre_reeval/mitre_reeval_PMC3969724",

    "../../mitre_reeval/mitre_reeval_PMC4047089",

    "../../mitre_reeval/mitre_reeval_PMC4122675",

    "../../mitre_reeval/mitre_reeval_PMC4423074",

    "../../mitre_reeval/mitre_reeval_PMC3547897",

    "../../mitre_reeval/mitre_reeval_PMC3378484",

    "../../mitre_reeval/mitre_reeval_PMC3329184",

    "../../mitre_reeval/mitre_reeval_PMC3284553",

    "../../mitre_reeval/mitre_reeval_PMC3250444",

    "../../mitre_reeval/mitre_reeval_PMC3102680",

    "../../mitre_reeval/mitre_reeval_PMC3056717",

    "../../mitre_reeval/mitre_reeval_PMC2982111",

    "../../mitre_reeval/mitre_reeval_PMC2964295",

    "../../mitre_reeval/mitre_reeval_PMC2920914",

    "../../mitre_reeval/mitre_reeval_PMC2920912",

    "../../mitre_reeval/mitre_reeval_PMC2900437",
]

if __name__ == "__main__":
    for curr_path in paths:
        pdi.run(amr_dot_file=curr_path, start_amr=0, end_amr=0, model_file='', passage_or_pmc='', entities_info_file='')

