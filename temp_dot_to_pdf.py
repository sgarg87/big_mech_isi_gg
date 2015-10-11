import pydot as pd
import textwrap as tw


# file_name = 'temp_files/pmid-24651010_pmid_2465_1010.144.dot'
file_name = 'temp_files/aaai_sdg.1.dot'
dot_graph = pd.graph_from_dot_file(file_name)
# sentence = dot_graph.get_label()
# dot_graph.label(tw.fill(sentence, 80))
file_name = file_name.replace('.', '_')
# dot_graph.write_pdf(file_name+'.pdf')
# dot_graph.write_svg(file_name+'.svg')
dot_graph.write_png(file_name+'.png')
# dot_graph.write_jpeg(file_name+'.jpeg')

