from pymongo import MongoClient
from libtopic_nlp_iit.db_util import *
import pandas as pd
import sys, os

def main(argv):
    BULK_SIZE = 500
    client = MongoClient('localhost', 27017)#['test-database']
    db = client["query_focus"]
    csv_scanner = pd.read_csv(argv[0], chunksize=BULK_SIZE)
    qf_name, _ = os.path.splitext(argv[0].split('/')[-1])
    # dict_webid2index = {t[1]: t[0] for t in enumerate(pd.read_csv(argv[0])["web_id"])}
    cur_index = 0 # mongodb does not secure document order; manual order index is specified
    for df_chunk in csv_scanner:
        docs = []
        for i in range(df_chunk.shape[0]):
            docs.append(validate_doc(df_chunk.iloc[i], order_index=cur_index+i))
        insert_doc_list(db, qf_name, docs)
        cur_index += df_chunk.shape[0]


if __name__ == "__main__": #python import_qf_csv.py kekkon.csv
    main(sys.argv[1:])
