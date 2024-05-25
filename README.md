# qdrant-test 
Rag sample with qdrant cloud and langchain and openai 

## prepare in advance
- create account openai and qdrant cloud
- qdrant apikey
- qdrant cloud url
- openai apikey

## in script
1. connect to qdrant cloud with apikey and endpoint 
2. check if collection is already exist and if not create collection
3. load date from url and insert qdrant with embeddings
4. get embeddings and docs from qdrant with question embedding
5. create prompt and create answer from openai 
