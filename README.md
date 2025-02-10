<h1 align="center"> SLAG <br> Enhancing LLMs for Expert Question Answering by Synergizing <br> with Knowledge Graphs </h1>

**SLAG** a QA framework for expert fields by **S**ynergizing large **L**anguage models **A**nd knowledge **G**raphs. It is designed to improve the correctness of KG retrieval and to intelligently coordinate KG retrievers with other retrieval methods through bidirectional enhancement between LLMs
and KGs:
* a LLM-Enhanced KG retriever is proposed to resolve entity ambiguities in queries and to accurately extract pertinent sub-graphs.
* a KG-Enhanced LLM reasoner is designed to distill sub-graphs and to flexibly generate answers or explicit queries for cooperative retrievers.

![SLAG](images/overview.png)

Since the privacy policy of our company, we only provide an open-source version of SLAG on three public datasets, 2wikimultihopqa, musique and hotpotqa, where the knowledge graphs are extracted with OpenIE. The experiment and performance can be reproduced for the public datasets, while the experiments on the financial datasets is not inluded in this repo due to the privacy of the used financial knowledge graph.  However, it is quite easy to deploy SLAG on a domain profession knowledge graph with strict schema, with a little code modification. 

## Setup Environment

Create a conda environment and install denpendency

```shell
conda create -n SLAG python=3.9
conda activate SLAG
pip install -r requirements.txt
```

Download and setup Neo4j Desktop or Docker, we provide a docker command here.
```shell
docker run \
  -p 7474:7475 -p 7687:7688 \
  --name multihop-public \
  -e "NEO4J_AUTH=neo4j/neo4jneo4j" \
  -e NEO4J_apoc_export_file_enabled=true \
  -e NEO4J_apoc_import_file_enabled=true \
  -e NEO4J_apoc_import_file_use__neo4j__config=true \
  -e NEO4J_PLUGINS=["apoc"] \
  neo4j:5.20.0
```

Download and setup Milvus store. We provide a docker command here
```shell
# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start the Docker container
$ bash standalone_embed.sh start
```

## Configuration
The configuration include the LLM api config, Neo4j database config and Milvus store config. We prefer to use **dotenv** package to load config into environment. You can create an .env file in the root of your program and write like this:
```Shell
OPENAI_API_BASE=your.openai.base
OPENAI_API_KEY=your.openai.key
OPENAI_USER_NAME=your.username

NEO4J_URI=your.neo4j.uri
NEO4J_USERNAME=your.neo4j.username
NEO4J_PASSWORD=your.neo4j.password
NEO4J_DATABASE=your.neo4j.database_name

MILVUS_URI="http://password@your.milvus.uri:your.port"
```

## Prepare Data

Load knowledge graph backup into Neo4j database. The knowlegde graph is extracted from raw corpus with GPT-3.5 and OpenIE prompt, which is referred to [HippRAG](https://github.com/OSU-NLP-Group/HippoRAG.git). The backup is stored at ./data/mhkg.dump. Using the following command:
```shell
neo4j-admin load --from=./data/mhkg.dump --database=your_database --force
```

Vectorize knowledge graph by embedding all the names of nodes into Milvus store with the following command:
```shell
python src/vectorize_kg.py
```

Chunking and vectorize corpus for RAG retriever with the following command:
```shell
python src/vectorize_rag.py
```

## Running SLAG

The main fuction is at `src/run_slag.py`. We provide two running mode, where **batch** is used to batch running slag on a dataset, and **server** is runnning a app with langserve to make single test and check intermediate steps.

### batch run mode
running command:
```shell
python src/run_slag.py --dataset_name 2wikimultihopqa --run_mode batch --use_llm_review
```
The batch mode will generate a result for the dataset, which is stored at `outputs/${dataset_name}/{dataset_name}_with_newquery_dataset.json`.

### server run mode
running command:
```shell
python src/run_slag.py --dataset_name 2wikimultihopqa --run_mode server --use_llm_review
```
This command will start a web app using langserve. Visit `http://your_ip_address:7105/slag/playground` and type the question in the corresponding dataset for single test.


## Evaluate SLAG



### Evaluation in RAG

Compare the performance of three methods including Naive RAG, Hybrid RAG and SLAG on three public QA datasets:
```shell
# 2wikimultihopqa
python src/run_eval_2wikimultihopqa.py

# hotpotqa
python src/run_eval_hotpotqa.py

# musique
python src/run_eval_musique.py
```

Detailed scores are stored by aim. You can just run ```aim up``` to launch the web UI.