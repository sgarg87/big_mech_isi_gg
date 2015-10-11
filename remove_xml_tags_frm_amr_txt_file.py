from lxml import etree


def remove_xml_tags(file_path):
    tree = etree.parse(file_path)
    notags = etree.tostring(tree, encoding='utf8', method='text')
    #
    with open(file_path+'_no_xml', 'w') as f:
        f.write(notags)


if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    remove_xml_tags(file_path)
