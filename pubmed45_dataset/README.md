This module avails the pubmed45 datatset, that has been manually annotated by authors containing approx. 26k interactions (approx. 2k positive labels). This is the primary datset used in the paper. This dataset has been acknowledged an one of important contributions in the paper since there is no previous publicly available dataset that has information on interaction type of interaction proteins. Previous annotated datasets only has a pair of interacting proteins as an interaction unit. Our interaction representation is richer as explained in the paper. This is why we had to manually annoatate this dataset. Here it is important to note that the dataset DO NOT use any sort of distant supervision techniques for annoatation and comes from pure manual annotation by single person (Sahil Garg, later is important since there can be disagreements between annoators otherwise). Though, some smart internal tools were used for the manual annoatation. All of this is important since it is manually labeled datasets that are best suitable for tesing of an extraction/classification algorithm while one can should use distant supervised like datasets only for training purposes (using distant supervised datsets for testing can lead to inflated F1 scores).

For an estimation for future annotators, annotation speed was approx. 100 interactions per hour. So, it took approx. 150 hours to annotate (1 months).


Format specification of JSON file, "pubmed45_data.json".

It  is a list of interactions where eac interaction is represented as a map with following fields.

path: 

path is more of an artificial path for other users who use this dataset. These paths were generated as per some local paths. However, there is also information on paper id like "pmid_1003_7796" in "../../eval-auto-amr-1000papers_june2015/eval-auto-amr/eval-auto-amr_a_pmid_1003_7796.11.dot_joint_0". These paper id like information can be used to divide datset into subsets for evaluation purpose. Later, in this document, we provide of subset listings we used in the AAAI16 paper.

interaction_tuple: 

This is the field which has a candidate interaction. It is basically a list of strings. The first value is always interaction type and the second one is catalyst. If length of the list is 3, the third entity is the one which is effected in the interaction catalyze by the catalyst (the second value). If length of the list is 4, then it is a complex-formation type interaction. In such case, third and fourth values represent the entities which are binding/dissociating from each other(or something that involves two entities besides the catalyst). For example, in the first interaction tuple below, it says that "CXCR4" stimulates "CCR5". In the second interaction tuple, it says that CCR5 catalyzes association of RANTES and FAK. Note that these examples are just candidate interactions tuples and not necessarily valid. Validity information is captured in the label field.
"interaction_tuple": [
            "stimulate", 
            "CXCR4", 
            "CCR5"
        ]	
"interaction_tuple": [
            "associate", 
            "CCR5", 
            "RANTES", 
            "FAK"
        ]  

label: 

There are three possible values for label: 0, 1, 2. Value 1 means that this interaction tuple is valid. Value 0 means it is invalid. Value 2 means that a swap between entity roles (i.e. swapping of catalyst role with one of the other two proteins) can make the interaction valid. This later label type can be useful for assembly task where half-valid interactions can also be useful in assemblying a biopathway model. Though, it make the classification task much harder since there is going to be a high correlation between samples from class 1 and 2. For example, in the second interaction tuple above, its label is 2 originally. But, swapping of "CCR5" with "RANTES" would give a valid interaction, i.e. "RANTES" becoming a catalyst and "CCR5" as a associating protein. 

sentence:
This field, as the name suggests, is source sentence from which the interaction is extracted.

 
