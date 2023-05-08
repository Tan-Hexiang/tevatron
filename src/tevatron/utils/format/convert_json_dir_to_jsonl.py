import thx.jsonl 
import json
from argparse import ArgumentParser
import pathlib
parser = ArgumentParser()
parser.add_argument('--json_dir', type=str, required=True, help='txt files')
args = parser.parse_args()

dir = pathlib.Path(args.json_dir)
files = dir.glob('*.json')
for file in files:
    output = str(file.parent)+'/'+file.stem+'.jsonl'
    print("read from {} ; write to {}".format(str(file),output))
    with file.open('r') as f:
        data = json.load(f)
        thx.jsonl.dump_all_jsonl(data,output)
