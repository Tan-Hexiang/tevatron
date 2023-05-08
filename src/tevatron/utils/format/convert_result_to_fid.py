from argparse import ArgumentParser
from datasets import load_dataset
import json
import src.thx.jsonl
import tqdm
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='txt files')
parser.add_argument('--output', type=str, required=True, help="jsonl files")
parser.add_argument('--save_ctxs_text', action='store_true')
parser.add_argument('--dataset_name', type=str, required=True)
parser.add_argument('--corpus_name', type=str, default=None)
parser.add_argument('--dataset_split', type=str)
parser.add_argument('--depth',type=int,default=100,required=True,help='与input文件中的depth对应')
args = parser.parse_args()

info = args.dataset_name.split('/')
args.dataset_split = info[-1] if len(info) == 3 else 'train'
args.dataset_name = "/".join(info[:-1]) if len(info) == 3 else '/'.join(info)

# load corpus
if args.save_ctxs_text and args.corpus_name:
    corpus = load_dataset(args.corpus_name)['train']
# load datasets
datasets = load_dataset(args.dataset_name)[args.dataset_split]
datasets = pd.DataFrame(datasets)

example = {}
result = []
# load result txt
with open(args.input, 'r') as in_f:
    for line in tqdm.tqdm(in_f):
        qid, docid, score = line.split()
        # question = datasets[int(qid)]['query']
        # answers = datasets[int(qid)]['answers']
        question = datasets.loc[datasets['query_id'] == qid].iloc[0]['query']
        answers = datasets.loc[datasets['query_id'] == qid].iloc[0]['answers']
        if example=={}:
            example = {
                'id':qid,
                'question':question,
                'answers':answers,
                'ctxs':[
                    {
                        'id':docid,
                        'score':score
                    }
                ]
            }
            if args.save_ctxs_text:
                example['ctxs'][0]['text'] = corpus[int(docid)-1]['text']
                example['ctxs'][0]['title'] = corpus[int(docid)-1]['title']

        elif int(example['id']) == int(qid):
            # 加入
            # assert int(qid) == int(example['id'])
            ctxs = {'id':docid,'score':score}
            if args.save_ctxs_text:
                ctxs['text'] = corpus[int(docid)-1]['text']
                ctxs['title'] = corpus[int(docid)-1]['title']
            example['ctxs'].append(ctxs)
            if len(example['ctxs']) == args.depth:
                # src.thx.jsonl.dump_jsonl_with_f(example,out_f)
                result.append(example)
                example = {}
        else:
            print("error !")
            exit()
            
    # json.dump(result,out_f)
    src.thx.jsonl.dump_all_jsonl(result,args.output)




'''
FID data format:
{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "id": 1234,
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "id": 4567,
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}

Tevatron Dataset format:
{'query_id': '0',
 'query': 'who got the first nobel prize in physics',
 'answers': ['Wilhelm Conrad Röntgen'],
 'positive_passages': [],
 'negative_passages': []}

Tevatron corpus data format:
 {'docid': '1',
 'text': 'Aaron Aaron ( or ; "Ahärôn") is a prophet, high priest, and the brother of Moses in the Abrahamic religions. Knowledge of Aaron, along with his brother Moses, comes exclusively from religious texts, such as the Bible and Quran. The Hebrew Bible relates that, unlike Moses, who grew up in the Egyptian royal court, Aaron and his elder sister Miriam remained with their kinsmen in the eastern border-land of Egypt (Goshen). When Moses first confronted the Egyptian king about the Israelites, Aaron served as his brother\'s spokesman ("prophet") to the Pharaoh. Part of the Law (Torah) that Moses received from',
 'title': 'Aaron'}
'''