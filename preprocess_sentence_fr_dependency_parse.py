def postprocess_sentence_frm_dependency_graph_parse(curr_sentence):
    # inner function
    def strip_quotes_spaces(x):
        if x.startswith("`") or x.endswith("`"):
            x = x.strip("`")
            is_strip = True
        elif x.startswith("'") or x.endswith("'"):
            x = x.strip("'")
            is_strip = True
        elif x.startswith(" ") or x.endswith(" "):
            x = x.strip(" ")
            is_strip = True
        else:
            is_strip = False
        return is_strip, x

    #main function start from here
    curr_sentence = curr_sentence.replace('\n', ' ')
    curr_sentence = curr_sentence.replace("\'", "").replace("\`", "").replace("\"", "") #.replace('.', '')
    curr_sentence = curr_sentence.replace('-LRB-', '(')
    curr_sentence = curr_sentence.replace('-RRB-', ')')
    curr_sentence = curr_sentence.replace(',', ', ')
    while True:
        is_strip, curr_sentence = strip_quotes_spaces(curr_sentence)
        if not is_strip:
            break
    return curr_sentence
