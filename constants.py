import re
import config_darpa as dc
import darpa_participant_types as dpt
import copy
import constants_locations_go_identifiers as clgi
from config_console_output import *
import config_interaction_type_extractions as cite


inhibit = "inhibit"
hyperphosphorylate = 'hyperphosphorylate'
dephosphorylate = 'dephosphorylate'
signal = 'signal'
activate = 'activate'
stimulate = 'stimulate'
mutate = 'mutate'
act = 'act'
hydrolyze = 'hydrolyze'
transcribe = 'transcribe'
stabilize = 'stabilize'
regulate = 'regulate'
downregulate = 'downregulate'
function = 'function'
apoptosis = 'apoptosis'
express = 'express'
potent = 'potent'
supress = 'suppress'
impede = 'impede'
potentiate = 'potentiate'
diminish = 'diminish'
enhance = 'enhance'
affect = 'affect'
increase = 'increase'
produce = 'produce'
decrease = 'decrease'
translocate = 'translocate'
localize = 'localize'
relocalize = 'relocalize'
recruit = 'recruit'
#
carboxymethylate = 'carboxymethylate'
depalmitoylate = 'depalmitoylate'
modulate = 'modulate'
delocalize = 'delocalize'
nitrosylate = 'nitrosylate'
ablate = 'ablate'
progress = 'progress'
senescence = 'senescence'
treat = 'treat'
loss = 'loss'
phenocopy = 'phenocopy'
# darpa list
phosphorylate = 'phosphorylate'
acetylate = 'acetylate'
deacetylate = 'deacetylate'
farnesylate = 'farnesylate'
glycosylate = 'glycosylate'
hydroxylate = 'hydroxylate'
methylate = 'methylate'
ribosylate = 'ribosylate'
sumoylate = 'sumoylate'
ubiquitinate = 'ubiquitinate'
myristoylate = 'myristoylate'
#
degrade = 'degrade'
#
translate = 'translate'
#
repress = 'repress'
replicate = 'replicate'
#
iodinate = 'iodinate'
deaminate = 'deaminate'
transactivate = 'transactivate'


if not dc.is_darpa:
    interaction_labels = [phosphorylate, dephosphorylate, hyperphosphorylate, activate, inhibit, signal, ubiquitinate, act,
                          hydrolyze, transcribe, stabilize, regulate, downregulate, function, apoptosis, express,
                          translocate, localize, relocalize, recruit, acetylate, deacetylate, farnesylate, glycosylate, hydroxylate,
                          methylate, ribosylate, sumoylate, carboxymethylate, depalmitoylate, modulate, delocalize,
                          nitrosylate, ablate, senescence, phenocopy, translate, repress, replicate]
    interaction_labels += [mutate, supress, impede, potentiate, diminish, enhance, increase, decrease, progress, treat,
                           loss]
    state_labels = copy.copy(interaction_labels)
else:
    interaction_labels = [phosphorylate, activate, inhibit, signal, ubiquitinate, act, transcribe, express, produce, potent, translocate,
                          localize, recruit, relocalize, acetylate, deacetylate, farnesylate, glycosylate, hydroxylate, hydrolyze,
                          methylate, ribosylate, sumoylate, delocalize, increase, decrease, degrade, translate, repress, replicate]

    chicago_interaction_types = [
            "liberate",
            "recruit",
            "conjugate",
            "react",
            "modulate",
            "release",
            "disrupt",
            "control",
            "increase",
            "block",
            "depend",
            "elevate",
            "modify",
            "alter",
            "deactivate",
            "attribute",
            "polymerize",
            "hyperphosphorylate",
            "regulate",
            "attenuate",
            "signal",
            "mutate",
            "methylate",
            "hydrolyze",
            "abrogate",
            "hinder",
            "demethylate",
            "iodize",
            "produce",
            "precede",
            "synergy",
            "upregulate",
            "result",
            "proteolyze",
            "hyperacetylate",
            "hypophosphorylate",
            "homolog",
            "translate",
            "discharge",
            "arrest",
            "synergize",
            "dephosphorylate",
            "require",
            "immunoprecipitate",
            "paralog",
            "generate",
            "degrade",
            "instigate",
            "polymerize",
            "deaminate",
            "constrain",
            "substitute",
            "prenylate",
            "diphosphorylate",
            "secrete",
            "function",
            "limit",
            "mannosylate",
            "inactivate",
            "n-glycosylate",
            "ubiquitinate",
            "transactivate",
            "splice",
            "ubiquitinize",
            "mediate",
            "signal",
            "overexpress",
            "sensitize",
            "glycosylate",
            "cut",
            "carbamylate",
            "free",
            "photoreactivate",
            phosphorylate,
            acetylate,
            deacetylate,
            methylate,
            hydrolyze,
            translate,
            dephosphorylate,
            degrade,
            hyperphosphorylate,
            #
            #
            ribosylate,
            deaminate,
            myristoylate,
            sumoylate,
            transcribe,
            activate,
            express,
            supress,
            produce,
            #
            activate,
            transactivate,
            regulate,
            translocate,
            'induce',
            'synthesize',
            stimulate,
            express,
            signal,
            phosphorylate,
            decrease,
            'immunoprecipitate',
            transcribe,
            inhibit,
            mutate,
            increase,
            ubiquitinate,
            'release',
            repress,
            'generate',
            potentiate,
            downregulate,
            'block',
            modulate,
            'elevate',
            'coimmunoprecipitate',
            'synergy',
            'suppress',
            'disrupt',
            'secrete',
            translate,
            iodinate,
            'coexpress',
            acetylate,
            produce,
            recruit,
            dephosphorylate,
            glycosylate,
            deaminate,
            'overexpress',
            degrade,
            'play',
            'polymerize',
            localize,
            'synergistic',
            'sensitize',
            'co-express',
            'sumoylate',
            'colocalize',
            'hydrolyze',
            'upregulate',
            'deacetylate',
            'exert',
            'separate',
            'coprecipitate',
            'methylate',
            'myristoylate',
            'deactivate',
            'relocalize',
            'hyperphosphorylate',
            'bond',
            'homologue',
            'replicate',
            'hypophosphorylate',
            'synergism',
            'potent',
            'co-immunoprecipitate',
            'repressor',
            'synergize',
            'ribosylate',
            farnesylate,
            'demethylate',
            'synergistically'
    ]
    #
    # interaction_labels += chicago_interaction_types
    # interaction_labels = list(set(interaction_labels))
    #
    state_labels = copy.copy(interaction_labels)
    state_labels = list(set(state_labels) - set([transcribe, express, translocate, localize, recruit, delocalize, increase, decrease]))
    state_labels += [mutate]
    location_labels = []
    location_labels += ['cytoplasm', 'membrane', 'plasma', 'plasma membrane', 'nucleus', 'extracellular', 'cell', 'nuclear', 'endoplasmic', 'apparatus']
    location_labels += clgi.GO_location_id_map.keys()
    state_labels += location_labels
    # interaction_labels += location_labels



bind = 'bind'
heterodimerize = 'heterodimerize'
dimerize = 'dimerize'
homodimerize = 'homodimerize'
form = 'form'
interact = 'interact'
dissociate = 'dissociate'
associate = 'associate'
complex = 'complex'
macro_molecular_complex = 'macro-molecular-complex'
micro_molecular_complex = 'micro-molecular-complex'
assemble = 'assemble'
disassemble = 'disassemble'


chicago_complex_types = [
        "couple",
        "separate",
        "break",
        "coprecipitate",
        "form",
        "bond",
        "attach",
        "cooperate",
        "unpair",
        "dissociate",
        "connect",
        "coimmunoprecipitate",
        "disengage",
        "independence",
        "copurify",
        "dissociate",
        "integrate",
        #
        bind,
        heterodimerize,
        dimerize,
        form,
        interact,
        dissociate,
        associate,
        complex,
        macro_molecular_complex,
        micro_molecular_complex,
        assemble,
        disassemble
]

if not dc.is_darpa:
    complex_labels = [bind, heterodimerize, form, dissociate, associate, complex, dimerize, homodimerize]
else:
    complex_labels = [bind, heterodimerize, form, dissociate, associate, complex, dimerize, homodimerize]
    #
    # complex_labels += chicago_complex_types
    # complex_labels = list(set(complex_labels))

protein = 'protein'
enzyme = 'enzyme'
small_molecule = 'small-molecule'
large_molecule = 'large-molecule'
molecule = 'molecule'
pathway = 'pathway'
cell = 'cell'
tumor = 'tumor'
cancer = 'cancer'
apoptosis = 'apoptosis'
gene = 'gene'
membrane = 'membrane'
nucleotide = 'nucleotide'
GTP = 'GTP'
GDP = 'GDP'
if not dc.is_darpa:
    protein_labels = [protein, enzyme, small_molecule, large_molecule, molecule, pathway, cell, complex, macro_molecular_complex, micro_molecular_complex, tumor, cancer, gene, membrane, nucleotide, GTP, GDP]
else:
    protein_labels = [protein, enzyme, small_molecule, large_molecule, molecule, complex, macro_molecular_complex, micro_molecular_complex, gene, dpt.chemical, dpt.protein_family]
    # protein_labels += [cell]

amino_acid = 'amino-acid'
protein_segment = 'protein-segment'
residue = 'residue'
peptide = 'peptide'
phosphorus = 'phosphorus'
state = 'state'

if not dc.is_darpa:
    protein_part_labels = [amino_acid, protein_segment, residue, peptide, phosphorus, state, small_molecule, large_molecule, molecule, cell, micro_molecular_complex, tumor, cancer, gene, pathway, GTP, GDP]
else:
    protein_part_labels = [amino_acid, protein_segment, peptide, cell, tumor, cancer, GTP, GDP, dpt.chemical]

all = 'all'
certain = 'certain'
this = 'this'

alphanum_regex = re.compile('[^a-zA-Z0-9]')
alpha_regex = re.compile('[^a-zA-Z]')
no_quotes_regexp = re.compile('^"|"|\'$')
#
concept_regexp = re.compile('[a-z]+[-][0-9][0-9]')
concept_num_regexp = re.compile('[-][0-9][0-9]')

list_of_invalid_state_or_interaction_type = \
    ['affect', 'seem', 'involve', 'mediate', 'direct', 'certain', 'responsible', 'serve', 'enhance', 'impact', 'contribute',
      'show', 'capable', 'suggest', 'result', 'consistent', 'lead', 'role', 'promote', 'elucidate', 'due', 'alter', 'become',
      'more', 'respond', 'both', 'level', 'change', 'reduce']

list_of_invalid_interaction_type = ['high', 'significant', 'mechanism', 'crucial', 'kinase']

# concept_labels = ['possible', 'use', 'require', 'correlate', 'prevent', 'inhibit', 'depend', 'respond', 'after', 'condition', 'cause', 'affect', 'result', 'disrupt']


NULL = 'NULL'
GARBAGE = 'GARBAGE'
state_change = 'state_change'
complex = 'complex'

#arguments of concept
ARG0 = 'ARG0'
ARG1 = 'ARG1'
ARG2 = 'ARG2'
ARG3 = 'ARG3'
value = 'value'
mod = 'mod'
condition = 'condition'

phosphoserine = 'phosphoserine'
active = 'active'
inactive = 'inactive'

part = 'part'
part_of = 'part-of'


state_change_dim = 6
complex_form_dim = 9

weight_round_decimal = 3


root = 'root'
