import time


def data_process(input_doc,tokenizer):

    samples = []
    with open(input_doc,encoding='utf-8') as f:
        lines = f.readlines()
    start_time = time.time()
    for i in range(1000):
        label_id = int(lines[i][-2])
        token_snet = [tokenizer.cls_token] + tokenizer.tokenize(lines[i][0:-3]) + [tokenizer.sep_token]
        sent_id = tokenizer.convert_tokens_to_ids(token_snet)
        samples.append([i, sent_id, label_id])
    end_time = time.time()
    print("读取数据的时间：", end_time - start_time, "秒")
    print(samples[-1])
    return samples #列表的列表，每个列表的第一个属性是行数，第二个属性是分词过后的文本对应的id，包含cls和sep，第三个属性是对应的标签