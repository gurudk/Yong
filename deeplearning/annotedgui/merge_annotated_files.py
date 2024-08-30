import glob
import datetime
import json

nowtime = datetime.datetime.now()
released_file = "./released/annotated.release." + nowtime.strftime("%Y%m%d%H%M%S") + ".txt"
files = glob.glob("./annotated/*.*.*.*")
all_dic = {}
with open(released_file, 'w') as wf:
    for file in files:
        with open(file, 'r') as infile:
            jsonobj = json.loads(infile.read())
            for key in jsonobj.keys():
                all_dic[key] = jsonobj[key]

    wf.write(json.dumps(all_dic))

print(len(all_dic))
