from asyncio.log import logger
import json

def dump_all_jsonl(data, output_path, append=True):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    # print('Wrote {} records to {}'.format(len(data), output_path))
    logger.info('Wrote {} records to {}'.format(len(data), output_path))

def dump_jsonl(data, output_path, append=True):
    """
    Write one objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json_record = json.dumps(data, ensure_ascii=False)
        f.write(json_record + '\n')
    # print('Wrote {} records to {}'.format(1, output_path))
    logger.info('Wrote {} records to {}'.format(1, output_path))

def dump_jsonl_with_f(data, f):
    """
    Write one objects to a JSON lines file.
    """
    json_record = json.dumps(data, ensure_ascii=False)
    f.write(json_record + '\n')
    # print('Wrote {} records to {}'.format(1, output_path))
    logger.info('Wrote {} records'.format(1))

def load_all_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    # print('Loaded {} records from {}'.format(len(data), input_path))
    logger.info('Loaded {} records from {}'.format(len(data), input_path))
    return data