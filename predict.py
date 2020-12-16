import json
import argparse
import nltk
import os
import pickle
from preprocess.utils import symbol_filter, re_lemma, fully_part_header, group_header, partial_header, num2year, group_symbol, group_values, group_digital
from preprocess.utils import AGG, wordnet_lemmatizer
from preprocess.utils import load_dataSets

import torch
from src import args as arg
from src import utils
from src.models.model import IRNet
from src.rule import semQL

from nltk.tokenize import word_tokenize
from sem2SQL import convert2SQL

class Model:
    def __init__(self, args, model, tables):
        self.args = args
        self.model = model
        self.tables = tables
        self.datas = []

def load_tables(args):
    with open(args.table_path, 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(table_datas)):
        table = table_datas[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        tabel_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table

    return tables

def create_datas(tables, db_id, question):
    datas = [{"db_id": db_id, "question": question, "question_toks": word_tokenize(question), "query": "SQL QUERY REDACTED"}]
    for d in datas:
        d['names'] = tables[d['db_id']]['schema_content']
        d['table_names'] = tables[d['db_id']]['table_names']
        d['col_set'] = tables[d['db_id']]['col_set']
        d['col_table'] = tables[d['db_id']]['col_table']
        keys = {}
        for kv in tables[d['db_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['db_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys
    return datas

def load_data(args):
    with open(args.table_path, 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    #with open(args.data_path, 'r', encoding='utf8') as f:
    #    datas = json.load(f)
    datas = [{"db_id": args.db_id, "question": args.question, "question_toks": word_tokenize(args.question), "query": "SQL QUERY REDACTED"}]

    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(table_datas)):
        table = table_datas[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        tabel_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table

    for d in datas:
        d['names'] = tables[d['db_id']]['schema_content']
        d['table_names'] = tables[d['db_id']]['table_names']
        d['col_set'] = tables[d['db_id']]['col_set']
        d['col_table'] = tables[d['db_id']]['col_table']
        keys = {}
        for kv in tables[d['db_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['db_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys
    return datas, tables

# pre-process input question
def process_datas(datas, args):
    """

    :param datas:
    :param args:
    :return:
    """
    print("Running process_datas...")
    with open(os.path.join(args.conceptNet, 'english_RelatedTo.pkl'), 'rb') as f:
        english_RelatedTo = pickle.load(f)

    with open(os.path.join(args.conceptNet, 'english_IsA.pkl'), 'rb') as f:
        english_IsA = pickle.load(f)

    # copy of the origin question_toks
    for d in datas:
        if 'origin_question_toks' not in d:
            d['origin_question_toks'] = d['question_toks']

    for entry in datas:
        entry['question_toks'] = symbol_filter(entry['question_toks'])
        origin_question_toks = symbol_filter([x for x in entry['origin_question_toks'] if x.lower() != 'the'])
        question_toks = [wordnet_lemmatizer.lemmatize(x.lower()) for x in entry['question_toks'] if x.lower() != 'the']

        entry['question_toks'] = question_toks

        table_names = []
        table_names_pattern = []

        for y in entry['table_names']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            table_names.append(" ".join(x))
            x = [re_lemma(x.lower()) for x in y.split(' ')]
            table_names_pattern.append(" ".join(x))

        header_toks = []
        header_toks_list = []

        header_toks_pattern = []
        header_toks_list_pattern = []

        for y in entry['col_set']:
            x = [wordnet_lemmatizer.lemmatize(x.lower()) for x in y.split(' ')]
            header_toks.append(" ".join(x))
            header_toks_list.append(x)

            x = [re_lemma(x.lower()) for x in y.split(' ')]
            header_toks_pattern.append(" ".join(x))
            header_toks_list_pattern.append(x)

        num_toks = len(question_toks)
        idx = 0
        tok_concol = []
        type_concol = []
        nltk_result = nltk.pos_tag(question_toks)

        while idx < num_toks:

            # fully header
            end_idx, header = fully_part_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for table
            end_idx, tname = group_header(question_toks, idx, num_toks, table_names)
            if tname:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["table"])
                idx = end_idx
                continue

            # check for column
            end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
            if header:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for partial column
            end_idx, tname = partial_header(question_toks, idx, header_toks_list)
            if tname:
                tok_concol.append(tname)
                type_concol.append(["col"])
                idx = end_idx
                continue

            # check for aggregation
            end_idx, agg = group_header(question_toks, idx, num_toks, AGG)
            if agg:
                tok_concol.append(question_toks[idx: end_idx])
                type_concol.append(["agg"])
                idx = end_idx
                continue

            if nltk_result[idx][1] == 'RBR' or nltk_result[idx][1] == 'JJR':
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MORE'])
                idx += 1
                continue

            if nltk_result[idx][1] == 'RBS' or nltk_result[idx][1] == 'JJS':
                tok_concol.append([question_toks[idx]])
                type_concol.append(['MOST'])
                idx += 1
                continue

            # string match for Time Format
            if num2year(question_toks[idx]):
                question_toks[idx] = 'year'
                end_idx, header = group_header(question_toks, idx, num_toks, header_toks)
                if header:
                    tok_concol.append(question_toks[idx: end_idx])
                    type_concol.append(["col"])
                    idx = end_idx
                    continue

            def get_concept_result(toks, graph):
                for begin_id in range(0, len(toks)):
                    for r_ind in reversed(range(1, len(toks) + 1 - begin_id)):
                        tmp_query = "_".join(toks[begin_id:r_ind])
                        if tmp_query in graph:
                            mi = graph[tmp_query]
                            for col in entry['col_set']:
                                if col in mi:
                                    return col

            end_idx, symbol = group_symbol(question_toks, idx, num_toks)
            if symbol:
                tmp_toks = [x for x in question_toks[idx: end_idx]]
                assert len(tmp_toks) > 0, print(symbol, question_toks)
                pro_result = get_concept_result(tmp_toks, english_IsA)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            end_idx, values = group_values(origin_question_toks, idx, num_toks)
            if values and (len(values) > 1 or question_toks[idx - 1] not in ['?', '.']):
                tmp_toks = [wordnet_lemmatizer.lemmatize(x) for x in question_toks[idx: end_idx] if x.isalnum() is True]
                assert len(tmp_toks) > 0, print(question_toks[idx: end_idx], values, question_toks, idx, end_idx)
                pro_result = get_concept_result(tmp_toks, english_IsA)
                if pro_result is None:
                    pro_result = get_concept_result(tmp_toks, english_RelatedTo)
                if pro_result is None:
                    pro_result = "NONE"
                for tmp in tmp_toks:
                    tok_concol.append([tmp])
                    type_concol.append([pro_result])
                    pro_result = "NONE"
                idx = end_idx
                continue

            result = group_digital(question_toks, idx)
            if result is True:
                tok_concol.append(question_toks[idx: idx + 1])
                type_concol.append(["value"])
                idx += 1
                continue
            if question_toks[idx] == ['ha']:
                question_toks[idx] = ['have']

            tok_concol.append([question_toks[idx]])
            type_concol.append(['NONE'])
            idx += 1
            continue

        entry['question_arg'] = tok_concol
        entry['question_arg_type'] = type_concol
        entry['nltk_pos'] = nltk_result

    return datas

def evaluate(args):
    """
    :param args:
    :return:
    """

    grammar = semQL.Grammar()
    val_sql_data, val_table_data= utils.load_predict_dataset(args.dataset, "serve/preprocessed.json", use_small=args.toy)

    model = IRNet(args, grammar)

    if args.cuda: model.cuda()

    print('load pretrained model from %s'% (args.load_model))
    pretrained_model = torch.load(args.load_model,
                                     map_location=lambda storage, loc: storage)
    import copy
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    model.word_emb = utils.load_word_emb(args.glove_embed_path)

    json_datas = utils.epoch_acc(model, args.batch_size, val_sql_data, val_table_data,
                           beam_size=args.beam_size)
    #print('Sketch Acc: %f, Acc: %f' % (sketch_acc, acc))
    # utils.eval_acc(json_datas, val_sql_data)
    import json
    with open('serve/prediction.json', 'w') as f:
        json.dump(json_datas, f)

def eval_inline(model, datas):
    print('eval...')
    json_datas = utils.epoch_acc(model.model, model.args.batch_size, datas, model.tables,
                           beam_size=model.args.beam_size)
    #print(f'-> {json_datas}')
    sql = convert2SQL(model, json_datas)
    return sql

def model_fn(model_dir):
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    args.conceptNet = 'preprocess/conceptNet'
    grammar = semQL.Grammar()
    model = IRNet(args, grammar)

    if args.cuda: model.cuda()

    print('load pretrained model from %s'% (model_dir))
    pretrained_model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    import copy
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    model.word_emb = utils.load_word_emb(args.glove_embed_path)
    tables = load_tables(args)
    return Model(args, model, tables)

def predict_fn(model, input):
    datas = create_datas(model.tables, input["db"], input["question"])
    preprocessed = process_datas(datas, model.args)
    #print(f'Preprocessed: {preprocessed}')
    sql = eval_inline(model, preprocessed)
    return sql

if __name__ == '__main__':
    model = model_fn('./saved_model/IRNet_pretrained.model')
    db = model.args.db_id
    while True:
        q = input(f'{db}> ')
        parts = q.split()
        if len(parts) > 1 and parts[0] == "db":
            db = parts[1]
            continue
        datas = create_datas(model.tables, db, q)
        preprocessed = process_datas(datas, model.args)
        #print(f'Preprocessed: {preprocessed}')
        sql = eval_inline(model, preprocessed)
        print(sql)

if __name__ == 'X__main__':
    arg_parser = arg.init_arg_parser()
    args = arg.init_config(arg_parser)
    args.conceptNet = 'preprocess/conceptNet'
    print(args)

    # loading dataSets
    datas, table = load_data(args) #load_dataSets(args)

    # process datasets
    process_result = process_datas(datas, args)
    print(f'process_result: {process_result}')

    with open(args.output, 'w') as f:
        json.dump(datas, f)

    evaluate(args)
    print('Done.')
