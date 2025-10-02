
def create_activity(provDoc, record):
    if len(provDoc.get_record(record))>0:
        raise Exception(f">create_activity({record}): a record already exists")
    return provDoc.activity(record)
    
def get_activity(provDoc,record):
    records = provDoc.get_record(record)
    if len(records)>0:
        return records[0]
    else:
        raise Exception(f">get_activity({record}): a record was not found") 