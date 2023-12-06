import numpy as np
from d3l.indexing.similarity_indexes import NameIndex, FormatIndex, ValueIndex, EmbeddingIndex, DistributionIndex

from d3l.input_output.dataloaders import CSVDataLoader
from d3l.input_output.dataloaders import PostgresDataLoader
from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object

# Postgres data loader

dataloader = PostgresDataLoader(
    db_host="localhost",
    db_password="postgres",
    db_username="postgres",
    db_name="postgres",
    db_port=5432
)



dataloader = CSVDataLoader(
    root_path='resources/data/',
    sep=','
)


# Create new indexes
name_index = NameIndex(dataloader=dataloader)
pickle_python_object(name_index, './name.lsh')
print("Name: SAVED!")

format_index = FormatIndex(dataloader=dataloader)
pickle_python_object(format_index, './format.lsh')
print("Format: SAVED!")

value_index = ValueIndex(dataloader=dataloader)
pickle_python_object(value_index, './value.lsh')
print("Value: SAVED!")

embedding_index = EmbeddingIndex(dataloader=dataloader, index_cache_dir="./")
pickle_python_object(embedding_index, './embedding.lsh')
print("Embedding: SAVED!")

distribution_index = DistributionIndex(dataloader=dataloader)
pickle_python_object(distribution_index, './distribution.lsh')
print("Distribution: SAVED!")


qe = QueryEngine(name_index, format_index, value_index, embedding_index, distribution_index)
results, extended_results = qe.table_query(table=dataloader.read_table(table_name='tableA'),
                                           aggregator=None, k=10, verbose=True)


        
data = [i.split(",")[1] for i in open("resources/data/student.csv",'r')]


# Name index query
results = name_index.query(query="price") # The query arg should be a column name. Tokenization will be performed automatically.
print(results)

# Format index query
results = format_index.query(query="<list/set>", k=10) # The query arg should be a collection of string values. The corresponding format descriptors will be extracted automatically.

# Value index query
results = value_index.query(query=data) # The query arg should be a collection of string values. Value pre-processing will be performed automatically.
print(results)

# Embeddings index query
results = embedding_index.query(query="<list/set>", k=10) # The query arg should be a collection of string values. The corresponding embeddings will be extracted automatically.

# Distribution index query
results = distribution_index.query(query="<list/set>") # The query arg should be a collection of numerical values. The corresponding distribution will be extracted automatically.