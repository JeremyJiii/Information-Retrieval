# -*- coding: utf-8 -*-
"""parse HTML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LW3uRxNp-_369h9DjQVRlWt2VLLbXEW7
"""

import sys
import json
import os
from bs4 import BeautifulSoup

sys.version
raw_dir = 'WEBPAGES_RAW/'
output_dir = 'Data/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def remove_unnecessary(name, soup):
    for s in soup.find_all(name):
        s.extract()

def add_to_dict(dictionary, name, tag, soup):
#     If nothing in the dictionary, add it.
    if(dictionary[name] == ''):
        dictionary[name] = []
    objs = soup.find_all(tag)
    if(len(objs) > 0):
        for obj in objs:
            s = obj.text.strip().replace('\n', ' ').replace('\t', ' ')
            if(len(s) > 0):
                dictionary[name].append(s)
            obj.extract()

def parse_data(soup, path, url):
    # Script and Style are meaningless
    remove_unnecessary('script', soup)
    remove_unnecessary('style', soup)

    # Define the dictionary content and the items in it.
    # Currently we define it as:
    '''
    Link_Name: the text show on link name, since in that link it is more relevant, we should store it here
    Paragraph: the paragraphs in the page, it is the important things
    Title: I just stored the h1, h2, h3, Strong tags here, they are more important content.
    Span: The ones in span, they are not important, I just store them and extract them later.
    Others: The other content left in the page, it should have less weight.
    '''
    content = {}
    content['Link_Name'] = []
    content['Paragraph'] = []
    content['Title'] = []
    content['Span'] = []
    content['Others'] = []
    content['url'] = url

    add_to_dict(content, 'Link_Name', 'a', soup)

    add_to_dict(content, 'Link_Name', 'li', soup)

    add_to_dict(content, 'Link_Name', 'option', soup)

    add_to_dict(content, 'Span', 'span', soup)

    add_to_dict(content, 'Paragraph', 'p', soup)

    for tag in ('title', 'h1', 'h2', 'h3', 'Strong'):
        add_to_dict(content, 'Title', tag, soup)
    try:
        soup.find('head').extract()
    except:
        None

#         Here since I extract all the read text above, the rest of them will not be duplicated.
#         And if there is any BROKEN html, it will read into "Others"
    for word in soup.text.split('\n'):
        word = word.strip()
        if(len(word) > 0):
            content['Others'].append(word)
            
    return content

def load_map_file(path):
    with open(path) as f:
        map_file = json.load(f)
    return map_file

# Initialize the parsing result
# It will delete what has alread been created!
# Don't run this one if it not necessary
def clean_result():
    map_file = load_map_file('bookkeeping.json')
    readed = {}
    for path in map_file:
        readed[path] = False
    with open('WEBPAGES_RAW/already_read.json', 'w') as f:
        json.dump(readed,f)
    for file_name in os.listdir(output_dir):
        os.remove(output_dir + file_name)

def read_and_write(max_time):
    total_result = {}
    try:
        with open('WEBPAGES_RAW/already_read.json') as f:
            alread_read = json.load(f)
    except:
        alread_read = {}

    map_file = load_map_file('WEBPAGES_RAW/bookkeeping.json')

    i = 0
    for path in map_file:
    #     If it is readed, do not read it again.
        if(path in alread_read.keys() and alread_read[path]):
            continue

        try:
            f = open('WEBPAGES_RAW/' + path)
        except:
            print('Something wrong with ' + path)
            continue
        else:
            soup = BeautifulSoup(f.read())
            f.close()
            url = map_file[path]
            total_result[path] = parse_data(soup, path, url)
        alread_read[path] = True
        i += 1
        if(i == max_time):
            break
    
    return total_result, alread_read

# clean_result()

dealed = 0
while(True):
    total_result, already_read = read_and_write(100)
    if(len(total_result) == 0):
        break

    data_name = str(len(os.listdir(output_dir))) + '.json'
    
#     Save data
    with open(output_dir + data_name, 'w') as f:
        json.dump(total_result, f)

#     Updata what has been readed.
    with open('WEBPAGES_RAW/already_read.json', 'w') as f:
        json.dump(already_read,f)
    dealed += len(total_result)











