
import torch 
from pathlib import Path
import logging
import json
import src.jsonl as jsonl
def print_grad_and_param(model,grad=True,para=True):
    print("=============更新之后===========")
    for name, parms in model.named_parameters():	
        print('-->name:', name)
        if para:
            print('-->para:', parms)
        if grad:
            print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("===")
# delete ctxs text
def rm_text(source_path, save_path):
    '''
    rm data[x]['ctxs'][x]['text'], x代表遍历
    source_path:.json or .jsonl
    save_path: .json or .jsonl, 会自动创建
    example:
        import src.thx
        src.thx.rm_text("/Users/tanhexiang/workplace/CFQA/CFQA/small_data/dpr_nq_result_100/dev87.jsonl","/Users/tanhexiang/workplace/CFQA/CFQA/small_data/dpr_nq_result_100_only_id/dev87.jsonl")
    '''
    save_path = Path(save_path)
    if not save_path.exists():
        save_path.parent.mkdir(parents=True,exist_ok=True)
        logging.info("create dir {}".format(str(save_path)))
    # load data
    source_path = Path(source_path)
    if source_path.suffix == ".jsonl":
        data = jsonl.load_all_jsonl(str(source_path))
    elif source_path.suffix == ".json":
        with open(str(source_path),'r') as f:
            data = json.load(f)
    else:
        logging.info("unkwon file suffix {}".format(source_path.suffix))
    # example的形式{..., 'ctxs':[{...,'text':"string"},{},{}]}
    # 此处迭代传引用，example中删除也对data起效,ctxs同理,以下两种写法均可以
    for example in data:
        # for key in range(len(example['ctxs'])):
        #     del example['ctxs'][key]['text']
        for ctxs in example['ctxs']:
            del ctxs['text']

    # re-save data
    if save_path.suffix == ".jsonl":
       jsonl.dump_all_jsonl(data,str(save_path))
    elif save_path.suffix == ".json":
        with save_path.open("w+") as f:
            json.dump(data,f) 
    else:
        logging.info("unknon file suffix {}".format(save_path.suffix))

# 计算top k accuracy
import logging
from src.evaluation import calculate_matches
def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    # logging.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits_acc = [v / len(data) for v in top_k_hits]
    # logging.info('Validation results: top k documents hits accuracy %s', top_k_hits_acc)
    # return match_stats.questions_doc_hits
    return top_k_hits,top_k_hits_acc
# 有text的文件转化为无text的文件
def rm_text_from_file(source_file,save_file):
    source = Path(source_file)
    if source.suffix == '.jsonl':
        data = jsonl.load_all_jsonl(str(source))
    elif source.suffix == '.json':
        with source.open('r') as f:
            data = json.load(f)
    else:
        logging.info("unexpected source file suffix {}".format(source.suffix))

    for example in source:
        del example['ctxs']['text']
    logging.info("Data[3]['ctxs'][5]:  {}".format(data[0]['ctxs'][0]))

    # save data
    save = Path(save_file)

# sample小样本数据
def sample_small_data_from_dir(data_dir,save_dir,ratio=0.1):
    # data_dir中的data应该是list形式
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True,exist_ok=True)
        logging.info("create dir {}".format(save_dir))
    logging.info(str(list(data_dir.glob('*.json*'))))
    for file in list(data_dir.glob('*.json*')):
        logging.info("sample data file from {}".format(file))
        if file.suffix=='.jsonl':
            data = jsonl.load_all_jsonl(str(data_dir))
        elif file.suffix=='.json':
            with file.open('r') as f:
                data = json.load(f)
        else:
            logging.info("unexpected file suffix {}".format(file.suffix))
        # logging.info("save data num {}".format(int(len(data)*ratio)))
        new_data = data[:int(len(data)*ratio)]
        logging.info("data len {}, sampled data len {}".format(len(data),len(new_data)))
        sampled_file = str(save_dir/(file.stem+str(int(len(data)*ratio))+'.jsonl'))
        jsonl.dump_all_jsonl(new_data,output_path=sampled_file)
        logging.info("Sampled data saved in {}".format(sampled_file))
    return

# 合并得到CF_score
import tqdm
def get_CF_score(pdist_list, loop_list):
    pdist = torch.Tensor(pdist_list)
    loop = torch.Tensor(loop_list)
    softmax = torch.nn.Softmax()
    # 先想加再取softmax
    # cf = softmax(pdist+loop)
    cf =pdist+loop
    cf_list = cf.tolist()
    return cf_list

def merge_score(adist_data, pdist_data, loop_data, save_path, n_context = 10, example_path ="example.jsonl"):
    new_data = []
    for k,example in enumerate(tqdm.tqdm(adist_data)):
        assert loop_data[k]['question']==pdist_data[k]['question'] and pdist_data[k]['question'] == example['question']
        example['LOOP_score'] = loop_data[k]['LOOP_score']
        example['PDist_score'] = pdist_data[k]['PDist_score']
        example['ADist_score'] = [ example['ctxs'][x]['score'] for x in range(n_context)]
        cf_list = get_CF_score(example['PDist_score'], example['LOOP_score'])
        example['CF_score'] = cf_list
        for i in range(n_context):
            assert example['ctxs'][i]['id'] == pdist_data[k]['ctxs'][i]['id'] and pdist_data[k]['ctxs'][i]['id'] == loop_data[k]['ctxs'][i]['id']
            example['ctxs'][i]['ADist_score'] = example['ctxs'][i]['score']
            example['ctxs'][i]['PDist_score'] = pdist_data[k]['ctxs'][i]['PDist_score']
            example['ctxs'][i]['LOOP_score'] = loop_data[k]['ctxs'][i]['LOOP_score']
            example['ctxs'][i]['CF_score'] = cf_list[i]
        new_data.append(example)
    jsonl.dump_all_jsonl(new_data , save_path, append=False)
    jsonl.dump_all_jsonl(new_data[:10], example_path, append=False)


if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s[%(pathname)s][line:%(lineno)d] - %(levelname)s: \n\t%(message)s',  level=logging.DEBUG)
    sample_small_data_from_dir(data_dir='/data/tanhexiang/CF_QA/small_data/dpr_result_only_id',save_dir='/data/tanhexiang/CF_QA/small_data/small',ratio=0.001)

