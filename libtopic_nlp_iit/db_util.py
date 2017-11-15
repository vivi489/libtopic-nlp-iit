from collections import defaultdict, deque
import pymongo
import pandas as pd
import json, ast, subprocess

def validate_doc(row, order_index=None): #get ready for db insertion
    doc = row.to_dict()
    doc["web_id"] = int(doc["web_id"])
    if doc["content"][-1] == '\n':
        doc["content"] = doc["content"][:-1]
    doc["entry"] = " ".join(ast.literal_eval(doc["entry"]))
    doc["suggest_ranks"] = json.loads(doc["suggest_ranks"])
    doc["topic"] = int(doc["topic"])
    doc["topic_probability"] = float(doc["topic_probability"])
    if order_index is not None: doc["index"] = order_index
    return doc

def insert_doc(db, coll_name, doc): #doc: a db-ready dict - keys: ['web_id', 'url', 'content', 'entry', 'suggests', 'suggest_ranks', 'topic', 'topic_probability']
    db[coll_name].create_index([('web_id', pymongo.ASCENDING)], unique=True)
    return db[coll_name].insert_one(doc).inserted_id

def insert_doc_list(db, coll_name, docs): #qf_name is the collection name
    db[coll_name].create_index([('web_id', pymongo.ASCENDING)], unique=True)
    return db[coll_name].insert_many(docs).inserted_ids

def dbcursor2df(dbcursor):
    dict_df = defaultdict(list)
    for doc in dbcursor:
        for k, v in doc.items():
            dict_df[k].append(v)
    return pd.DataFrame(dict(dict_df))

def get_topN(db, qf_name, N):# return a frame
    n_topics = db[qf_name].find_one({}, {"topic": 1, '_id': False},
                                    sort=[('topic', pymongo.DESCENDING)])["topic"] + 1
    list_df_t = []
    for t in range(n_topics):
        dbcursor = db[qf_name].find({"topic": t}, {'_id': False}).sort('topic_probability', pymongo.DESCENDING)[:N]
        list_df_t.append(dbcursor2df(dbcursor))
    return pd.concat(list_df_t)

def extract_one_segment_suggest_keywords(db, qf_name, qf_name_JPC): #retrieve set_valid_suggest_words for a query focus
    #return a set of keywords
    retVal = set()
    for doc in db[qf_name].find():
        suggests = doc["suggests"]
        for suggest in suggests.split('\n'):
            if len(suggest) == 0: continue
            suggest_tokens = set([w for w in suggest.split() if len(w)>0])
            if qf_name_JPC in suggest_tokens:
                suggest_tokens.remove(qf_name_JPC)
            if len(suggest_tokens)==1: retVal.update(suggest_tokens)
    return retVal

def tokenize_doc_collection(db, qf_name):
    q_wid = deque()
    with open("%s_temp"%qf_name ,'w') as f:
        for doc in db[qf_name].find({}, snapshot=True):
            q_wid.append(doc["web_id"])
            f.write(doc["content"])
            f.write('\n')
    subprocess.run(["mecab", "-b 200000", "-Owakati", "%s_temp"%qf_name, "-o", "%s_token_temp"%qf_name])
    with open("%s_token_temp"%qf_name ,'r') as f:
        for line in f:
            db[qf_name].update_one({"web_id": q_wid[0]}, {"$set": {"token": line}})
            q_wid.popleft()
    subprocess.run("rm ./%s_temp"%qf_name, shell=True)
    subprocess.run("rm ./%s_token_temp"%qf_name, shell=True)


