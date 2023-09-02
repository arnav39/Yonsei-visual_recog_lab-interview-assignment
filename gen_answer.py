from os import listdir
from os.path import isfile, join
from collections import OrderedDict

ordered_dic = OrderedDict()

mypath = './data/test_answer/positive'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
	ordered_dic[int(f)] = 1

mypath = './data/test_answer/negative'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
	ordered_dic[int(f)] = 0

ordered_dic = OrderedDict(sorted(ordered_dic.items(), key=lambda item: item))

print('id,sentiment')
for dict_item in ordered_dic:
	print(str(dict_item)+','+str(ordered_dic[dict_item]))