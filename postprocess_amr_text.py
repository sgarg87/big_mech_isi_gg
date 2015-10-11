

def post_process_amr_text_sentence(curr_sentence):
    curr_sentence = curr_sentence.replace('\n', '')
    curr_sentence = curr_sentence.replace(' @\-@ ', '\-')
    curr_sentence = curr_sentence.replace(' @-@ ', '-')
    curr_sentence = curr_sentence.replace(' @/@ ', ' / ')
    curr_sentence = curr_sentence.replace(' @:@ ', ' : ')
    curr_sentence = curr_sentence.replace('@- ', '-')
    curr_sentence = curr_sentence.replace(' -@', '-')
    curr_sentence = curr_sentence.replace('\\', '')
    curr_sentence = curr_sentence.replace('@\- ', '\-')
    curr_sentence = curr_sentence.replace(' \-@', '\-')
    curr_sentence = curr_sentence.replace('@', '')
    assert '@' not in curr_sentence, curr_sentence
    assert '\\' not in curr_sentence, curr_sentence
    return curr_sentence

