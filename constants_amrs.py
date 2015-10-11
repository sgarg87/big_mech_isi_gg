from config_console_output import *


default_amrs = []


bmtr_4 = [1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]
default_amrs.append({'path': './gold-amr-2015-01-08-ulf/bio-bmdr4-gold-2015-01-08_bio.bmtr_0004', 'idx': [1, 27], 'labels': bmtr_4})
bmtr_4 = None

# It has recently been shown that oncogenic RAS can enhance the apoptotic function of p53 via ASPP1 and ASPP2.
# Mechanistically ASPP1 and ASPP2 bind RAS-GTP and potentiates RAS signalling to enhance p53 mediated apoptosis [2].
# As RAS is upstream of several signalling cascades [13], we queried whether the activity of ASPP2 is regulated
# by the activation of a RAS-mediated signalling pathway. One of the most studied downstream pathways of RAS signalling
# is the Raf-MAPK pathway. Interestingly, we observed two conserved putative MAPK phosphorylation sites in ASPP1 and ASPP2.
# The ASPP1 sites are at residues 671 and 746, and the ASPP2 sites are at residues 698 and 827 (Figure 1A). We thus tested
# whether RAS activation may regulate ASPP2 phosphorylation. An in vitro phophorylation assay was performed with a purified
# C-terminus fragment of ASPP2 (693-1128) containing both MAPK putative phosphorylation sites. When compared to p38 SAPK, MAPK1
# was clearly able to phosphorylate the ASPP2 fragment in vitro (Figure 1B, left and middle panels). As shown in Figure S1, histone
# 2B phosphorylated by p38 SAPK had high levels of incorporated 32P, suggesting that p38 SAPK was active; while under the same
# conditions, ASPP2 (693-1128) fragment phosphorylated by p38 SAPK had very low levels of incorporated 32P, indicating that p38 SAPK
# is not an efficient kinase for ASPP2 phosphorylation. The phosphorylated ASPP2 fragment by MAPK1 was digested by trypsin and
# fractioned on a high performance liquid chromatography (HPLC). Each eluted fraction was measured for its radioactivity content
# (Figure 1B, right panel). The fractions representing these radioactive peaks were analysed by mass spectrometry. Of the two
# radioactive peaks, one represented the linker region between the GST and our ASPP2 fragment and the other corresponded to a
# fragment of the same mass as that containing the second putative phosphorylation site, serine 827. Hence ASPP2 can be phosphorylated
# at serine 827 by MAPK1 in vitro. We and others have recently shown that ASPP2 can potentiate RAS signaling by binding directly via the ASPP2 N-terminus [2,6].
# Moreover, the RAS-ASPP interaction enhances the transcription function of p53 in cancer cells [2]. Until now, it has been unclear
# how RAS could affect ASPP2 to enhance p53 function. We show here that ASPP2 is phosphorylated by the RAS/Raf/MAPK pathway and that
# this phosphorylation leads to its increased translocation to the cytosol/nucleus and increased binding to p53, providing an explanation
# of how RAS can activate p53 pro-apoptotic functions (Figure 5). Additionally, RAS/Raf/MAPK pathway activation stabilizes ASPP2 protein,
# although the underlying mechanism remains to be investigated.

bmtr_5 = [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1]
#skipping 12
default_amrs.append({'path': './gold-amr-2015-01-08-ulf/bio-bmdr4-gold-2015-01-08_bio.bmtr_0005', 'idx': [1, 11], 'labels': bmtr_5[0:11]})
default_amrs.append({'path': './gold-amr-2015-01-08-ulf/bio-bmdr4-gold-2015-01-08_bio.bmtr_0005', 'idx': [13, 23], 'labels': bmtr_5[12:23]})
bmtr_5 = None

#
# We utilized an unbiased mass spectrometry-based approach to identify ubiquitination sites of Ras. His-tagged ubiquitin
# and Flag-tagged K-Ras4B (K-Ras hereafter) were expressed in HEK293T cells at levels similar to endogenous K-Ras (Fig. 1B)
# and subjected to sequential affinity chromatography. His-ubiquitinated proteins were purified by Co2+ metal affinity
# chromatography in 8M urea denaturing conditions. His-ubiquitinated K-Ras was subsequently purified with anti-Flag resin.
# Following purification, mono- and di- ubiquitinated K-Ras appeared to be the major ubiquitination forms, which is
# consistent with the endogenous K-Ras ubiquitination pattern (Fig. 1, A and B). H-Ras ubiquitination sites were also
# determined by the same approach. Tandem mass spectrometric analysis of tryptic fragments from the bands migrating at
# the positions expected for mono- and di-ubiquitinated Ras revealed ubiquitination at Lys residues 104 and 147 of K-Ras,
# and Lys residues 117, 147 and 170 for H-Ras (fig. S1C). The tryptic peptide with ubiquitination at Lys147 (K147) was the
# most frequently observed peptide for both K-Ras and H-Ras, while Lys117 appeared as a secondary major ubiquitination site in H-Ras.
# To examine the effect of ubiquitination on GTP loading, we purified wild-type K-Ras, oncogenic G12V-K-Ras mutant or the ubiquitinated
# subfraction of wild-type K-Ras from 32P-orthophosphate labeled cells and utilized thin layer chromatography (TLC) and high performance
# liquid chromatography (HPLC) to assess the ratio of 32P-GTP to 32P-GDP that co-purified with each form of K-Ras. As expected based on
# previous studies, wild-type K-Ras bound primarily 32P-GDP, while G12V-Ras bound 32P-GTP (Fig.2, A and B). Interestingly, the
# ubiquitinated subfraction of wild-type K-Ras retained a significant amount of 32P-GTP. These results are consistent with a model
# in which ubiquitination of Lys147 (or Lys117), destabilizes GDP binding, allowing spontaneous GDP/GTP exchange. It could be argued
# that GTP loading occurs prior to ubiquitination and that the GTP bound form of K-Ras, via interaction with effectors, is preferentially
# mono-ubiquitinated via a feedback mechanism. While it is difficult to eliminate this possibility, it is unlikely since, as shown in
# fig. S1B, the T35 mutant of K-Ras, which fails to interact with downstream effectors (fig. S1B) undergoes comparable monobuiquitination
# to wild type Ras. These results, along with the crystal structure, support a model in which mono-ubiquitination at a Lys residue directly
# involved in GDP binding either enhances nucleotide exchange on K-Ras, impairs GTP hydrolysis, or both. To corroborate this finding,
# we measure the activity of Ras by the GST-RBD pull-down assay. To ensure that ubiquitinated Ras was being detected, the protein pulled
# down by GST-RBD was subjected to a second affinity purification on a cobalt column to purify the Flag-His-tagged K-Ras. As predicted,
# only a very small fraction of wild-type K-Ras was pulled down by the GST-RBD (Fig. 2C and fig. S1D), consistent with very little wild-type
# K-Ras being in the GTP state under these conditions (Fig.2, A and B). However, a much greater fraction of the ubiquitinated-K-Ras was
# pulled down by the GST-RBD (Fig. 2C and fig. S1D). These results are consistent with a greater fraction of ubiquitinated K-Ras being in
# the GTP state (Fig. 2, A and B).
#
bmtr_1 = [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1]
#skipping 9
default_amrs.append({'path': './bio-bmtr1/bio-bmtr1_bio.bmtr_0001', 'idx': [1, 8], 'labels': bmtr_1[0:8]})
#skipping 17, 18
default_amrs.append({'path': './bio-bmtr1/bio-bmtr1_bio.bmtr_0001', 'idx': [10, 16], 'labels': bmtr_1[9:16]})
default_amrs.append({'path': './bio-bmtr1/bio-bmtr1_bio.bmtr_0001', 'idx': [19, 21], 'labels': bmtr_1[18:21]})
bmtr_1 = None


# We hypothesized that the MEK/ERK pathway may suppress trans-phosphorylation of ERBB3 by directly phosphorylating the JM domains of
# EGFR and HER2, and that this could be a dominant MEK inhibitor-induced feedback leading to AKT activation in these cancers. We used
# tandem mass spectrometry to measure the effects of AZD6244 on phosphorylation of this JM domain threonine residue in both EGFR-mutant
# and HER2- amplified cancer models. Targeting both the phosphorylated and non-phosphorylated peptide forms, we detected a 66% average
# decrease in EGFR T669 phosphorylation and a 75% decrease in HER2 T677 phosphorylation upon treatment with AZD6244 (Figure 5B,
# Supplemental Figure 8). Phospho-specific antibodies confirmed that treatment with AZD6244 inhibited phosphorylation of T669 of EGFR
# and the analogous T677 of HER2 (Figure 5A). Together these data indicate that loss of this inhibitory threonine phosphorylation on
# the JM domains of EGFR and HER2 occurs in cancer cell lines following MEK inhibition, presumably due to differential subcellular
# localization and/or binding proteins. Mutation of T669 and T677 abrogates MEK inhibitor-induced suppression of ERBB3 Activation
# We hypothesized that MEK inhibition activates AKT by inhibiting ERK activity, which blocks an inhibitory threonine phosphorylation on
# the JM domains of EGFR and HER2, thereby increasing ERBB3 phosphorylation. To test this hypothesis, we transiently transfected CHO-KI
# cells, which do not express ERBB receptors endogenously, with wildtype ERBB3 with either wild-type EGFR or EGFR T669A. In cells
# transfected with wildtype EGFR, MEK inhibition led to feedback activation of phospho-ERBB3 and phosho- EGFR, recapitulating the results
# we had observed in our panel of cancer cell lines (Figure 6A). In contrast, the EGFR T669A mutant increased both basal EGFR and ERBB3
# tyrosine phosphorylation that was not augmented by MEK inhibition. As a control, we treated CHOKI cells expressing EGFR T669A with HRG
# ligand to induce maximal ERBB3 phosphorylation (Figure 6A), indicating that the lack of induction of phospho-ERBB3 in EGFR T669A
# expressing cells following MEK inhibition was not simply due to the saturation of the system with phospho-ERBB3. We observed analogous
# results in CHO-KI cells expressing wild-type ERBB3 in combination with wild-type or T677A mutant HER2 (Figure 6B). Together these
# results support the hypothesis that inhibition of ERK-mediated phosphorylation of a conserved JM domain threonine residue leads to
# feedback activation of EGFR, HER2, and ERBB3 (Figure 7).

bmtr_2 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1]
default_amrs.append({'path': './bio-bmtr1/bio-bmtr1_bio.bmtr_0002', 'idx': [1, 14], 'labels': bmtr_2})

#
# We identify four S/TP sites of B-Raf phosphorylated by activated ERK and find that feedback phosphorylation of B-Raf inhibits
# binding to activated Ras and disrupts heterodimerization with C-Raf, which is dependent on the B-Raf pS729/14-3-3 binding site.
# 14-3-3 dimers bind to phosphorylation sites present in both the N- and C-terminal regions and stabilize the autoinhibited state (22).
# To activate the Raf proteins, autoinhibition mediated by the N terminus must be relieved and the kinase domain must adopt the active
# catalytic conformation. Under normal signaling conditions, Ras activation helps mediate these events by recruiting the Raf
# proteins to the plasma membrane, which induces the release of 14-3-3 from the N-terminal binding site and facilitates
# phosphorylation of the Raf kinase domain (19). Once activated, either by upstream signaling or by mutational events, all Raf
# proteins are capable of initiating the phosphorylation cascade that results in the sequential activation of MEK and ERK. Strikingly,
# the Raf proteins themselves are also substrates of activated ERK. In regard to C-Raf, ERK-dependent feedback phosphorylation
# has been shown to instigate a regulatory cycle whereby phosphorylation of the feedback sites down-modulates C-Raf signaling,
# after which the hyperphosphorylated C-Raf protein is dephosphorylated and returned to a signaling-competent state through
# dephosphorylation events involving protein phosphatase 2A (PP2A) and the Pin1 prolyl-isomerase (8). For B-Raf, two ERK-dependent
# feedback sites, S750 and T753, have been identified, and phosphorylation of these sites has been reported to have a negative
# regulatory effect. Here we find that both normal and oncogenic B-Raf proteins are phosphorylated on four S/TP sites (S151,
# T401, S750, and T753) by activated ERK. Previously, we found that in response to growth factor treatment, signaling from C-Raf
# is downregulated by ERK-dependent feedback phosphorylation on S/TP sites and that C-Raf is subsequently dephosphorylated and
# returned to a signaling-competent state through the activities of PP2A and the Pin1 prolyl-isomerase (8). The Pin1 prolyl-isomerase
# binds specifically to phosphorylated S/TP (pS/TP) motifs (33), and isomerization of the pS/TP bond is required for PP2A to
# efficiently dephosphorylate certain proteins, such as cdc25C, Myc, and C-Raf (16). Complex formation between B-Raf and Pin1
# correlated with the phosphorylation of B-Raf on S/TP sites (Fig. 1C) and this interaction could be blocked when the MEK
# inhibitor U0126 was used to prevent ERK activation and the S/TP phosphorylation of B-Raf (Fig. 1D). Together, these findings
# indicate that Pin1 is needed for the efficient dephosphorylation of B-Raf and are consistent with the model that S/TP
# phosphorylation inhibits Raf signaling. Eluting in HPLC fractions 78 to 79 was a peptide phosphorylated on S750 and T753, the
# previously identified ERK sites, and eluting in fractions 26 and 58 to 59 were peptides phosphorylated at S151 and T401,
# respectively. All four of these identified sites are followed by a proline residue, and their phosphorylation could be blocked
# by pretreating cells with the MEK inhibitor U0126 (Fig. 2A), suggesting that these residues are feedback targets of the
# proline-directed kinase, ERK. Consistent with this model, we found that when purified activated ERK was incubated with
# kinase-dead B-Raf(K375M) in vitro, ERK strongly phosphorylated B-Raf on the S151, S750, and T753 sites, with phosphorylation
# of T401 also observed (Fig. 2B). These findings are similar to what has been observed for C-Raf (8) and suggest that feedback
# phosphorylation is a conserved mechanism used to disrupt the Ras/Raf interaction. Consistent with these data, we found that
# B-Raf interacted with C-Raf in an inducible and transient manner following growth factor treatment (Fig. 3B and C). In addition,
# when B-Raf feedback phosphorylation was prevented, either by U0126 treatment or by mutation of all the feedback sites, an
# increase in the basal level of heterodimerization with C-Raf was observed, and heterodimerization in response to growth factor
# treatment was increased and prolonged (Fig. 3B and C). These findings support a model whereby feedback phosphorylation disrupts
# Raf heterodimerization. Unlike WT B-Raf, oncogenic B-Raf proteins have been shown to heterodimerize constitutively with C-Raf
# in a Ras-independent manner (11). When we next examined the effect of feedback phosphorylation on the ability of oncogenic B-Raf
# to form heterodimers with C-Raf, we found that the levels of endogenous C-Raf associating with B-Raf proteins of high (V600E),
# intermediate (G466A), and impaired (D594G) kinase activities all increased when the feedback sites were mutated, indicating
# that feedback phosphorylation also inhibits the heterodimerization of oncogenic B-Raf proteins (Fig. 3D). Previous studies have
# shown that, for both normal and oncogenic B-Raf proteins to heterodimerize with C-Raf, the C-terminal 14-3-3 binding site of
# C-Raf (S621) must be intact (11, 27) (Fig. 3E). To determine whether binding of 14-3-3 to B-Raf is also required for hetero=
# -dimerization, B-Raf proteins containing lanine substitutions in the two 14-3-3 binding sites, S365 and S729 (2), were examined
# for their abilities to heterodimerize with C-Raf in response to growth factor treatment. Not surprisingly, given that mutation
# of the S365 14-3-3 binding site enhances the membrane localization of B-Raf (2), increased heterodimerization with C-Raf was
# observed for S365A B-Raf compared to WT B-Raf (Fig. 3F). In contrast, S729A B-Raf failed to heterodimerize with C-Raf in response
# to growth factor treatment, and mutation of this site disrupted the constitutive interaction of oncogenic B-Raf proteins and
# C-Raf (Fig. 3F), indicating that heterodimerization with C-Raf is dependent on the C-terminal S729 14-3-3 binding site of B-Raf.
# Given that oncogenic B-Raf proteins are targets of feedback phosphorylation, we next examined whether they might also be
# dephosphorylated and recycled in a manner involving the PP2A phosphatase and the Pin1 prolyl-isomerase. As indicated in Fig.
# 7A, when PP2A was inhibited with okadaic acid treatment, slower-migrating forms of the V600E, G466A, and D594G B-Raf proteins
# were found to accumulate. Moreover, given their constitutive phosphorylation on S/TP sites (Fig. 3D), these oncogenic B-Raf
# mutants were found to interact constitutively with Pin1 (Fig. 7B), indicating that oncogenic B-Raf proteins are dephospho-
# -rylated and recycled. Consistent with the model that Pin1 influences B-Raf signaling by facilitating the dephosphorylation of
# the feedback sites, overexpression of the Pin1 proteins had no effect on the transformation potential of G466A FBm-B-Raf,
# which lacks the sites of feedback phosphorylation. Previous studies have found that both the C-Raf and B-Raf proteins are
# targets of ERK-dependent feedback phosphorylation. In the case of C-Raf, six sites of feedback phosphorylation have been
# identified, five of which are direct targets of activated ERK (8). For B-Raf, previous work by Brummer et al. (3) identified
# the C-terminal S750 and T753 residues as sites phosphorylated by activated ERK. Through metabolic labeling experiments, we
# find here that in addition to the S750 and T753 sites, B-Raf is feedback phosphorylated on two other sites, S151 and T401.
# These residues are phosphorylated by activated ERK in vitro, As has been observed for C-Raf, we find that the hyper-
# -phosphorylated B-Raf protein is subsequently dephosphorylated in a manner requiring the activities of the PP2A phosphatase
# and Pin1 prolyl-isomerase, indicating that the feedback phosphorylation/dephosphorylation cycle is a conserved regulatory
# mechanism for the Raf proteins. Taken together, these findings suggest a model whereby the binding of a 14-3-3 dimer to the
# C-terminal pS621 site of C-Raf and the C-terminal pS729 site of B-Raf provides the stable docking event that then allows the
# two proteins to make additional contacts (Fig. 9).

mskcc_1_26 = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
default_amrs.append({'path': './bio-mskcc/bio-mskcc-57-40/bio-mskcc-57-40_bio.mskcc_0001', 'idx': [1, 6], 'labels': mskcc_1_26[0:6]})
default_amrs.append({'path': './bio-mskcc/bio-mskcc-57-40/bio-mskcc-57-40_bio.mskcc_0001', 'idx': [8, 26], 'labels': mskcc_1_26[7:26]})
mskcc_1_26 = None

mskcc_43_52 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
default_amrs.append({'path': './bio-mskcc/bio-mskcc-57-40/bio-mskcc-57-40_bio.mskcc_0001', 'idx': [44, 48], 'labels': mskcc_43_52[1:6]})
default_amrs.append({'path': './bio-mskcc/bio-mskcc-57-40/bio-mskcc-57-40_bio.mskcc_0001', 'idx': [50, 51], 'labels': mskcc_43_52[-3:-1]})
mskcc_43_52 = None

mskcc_57 = [1]
default_amrs.append({'path': './bio-mskcc/bio-mskcc-57-40/bio-mskcc-57-40_bio.mskcc_0001','idx': [57, 57], 'labels': mskcc_57})
mskcc_57 = None




