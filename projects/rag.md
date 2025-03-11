# RAG

## qdrant

```
docker
run -d -p 6333:6333 \
    -v $(pwd)/data:/qdrant/storage \
    -v $(pwd)/snapshots:/qdrant/snapshots \
    -v $(pwd)/config.yaml:/qdrant/config/production.yaml \
    qdrant/qdrant
```

config.yaml 
https://github.com/qdrant/qdrant/blob/master/config/config.yaml

