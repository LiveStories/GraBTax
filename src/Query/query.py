
from hermes.Search.IndicatorSearch.Queries.predict_keywords import predict_keywords
from elasticsearch import Elasticsearch, RequestsHttpConnection
from networkx import Graph
from hermes.Config import configmap
from hermes.Search.SearchProcessing import process
from hermes.TopicModelling import LDA
from pithos.Utils import stringutils
import os
import GraBTax.Subgraph.build_graph as graph
import csv
import boto3
from requests_aws4auth import AWS4Auth
from distutils.util import strtobool


session = boto3.Session()
credentials = session.get_credentials()
config = configmap.ConfigMap()
model_path = config.section_map("Models")["model_path"]
environment = config.section_map("SearchSettings")["environment"]
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, "us-west-2", 'es', session_token=credentials.token)


es = Elasticsearch(hosts=[{"host": config.section_map(environment)["host"],
                           "port":int(config.section_map(environment)["port"])}],
                   http_auth=awsauth, use_ssl=bool(strtobool(config.section_map(environment)["use_ssl"])),
                   verify_certs=bool(strtobool(config.section_map(environment)["verify_certs"])),
                   connection_class=RequestsHttpConnection)


with open(os.path.join(model_path, "category_guesses_cleaned.csv"), "r") as infile:
    reader = csv.reader(infile, delimiter=",")
    topic_words = {int(rows[0]): rows[2] for rows in reader}


def get_topic_clause(topics):
    clauses = []
    for topic in topics:
        clauses.append({
            "term": {
                "topics": topic
            }
            }
        )
    return clauses


def get_body(topics):
    json = {
        "query": {
            "bool": {
                "must": get_topic_clause(topics),
                "must_not": [],
                "should": []
            }
        },
        "from": 0,
        "size": 10,
        "sort": [],
        "aggs": {}
    }
    return json


def traverse_graph(g, head, topic_list=None):
    if topic_list is None:
        topic_list = []

    def check_topic(topic):
        if topic in g.edge["query"].keys():
            if topic not in topic_list:
                return True
            else:
                return False
        return True

    if head in g.edge["query"].keys():
        if head not in topic_list:
            topic_list = []

    for child in g.edge[head]:
        # todo: better way of ignoring indicator vertex than ignoring str (we only want topic vertices)
        if child != "query" and child != head and check_topic(child) and type(child) != str:
            topic_list.append(child)
            results = search_indicators_by_topic(topic_list)
            print(child)
            for hit in results["hits"]["hits"]:
                id_ = hit["_source"]["id"]
                score = hit["_score"]
                g.add_edge(child, id_, weight=score)
            traverse_graph(g, child, topic_list)
    return g


def search_indicators_by_topic(topics):
    results = es.search("liq_indicators_b", doc_type="indicator", body=get_body(topics))
    return results


def query(query_text):
    tokenized_sentences = stringutils.tokenize(query_text)
    for word in tokenized_sentences:
        if word not in LDA.lda.id2word.token2id.keys():
            probable_word = predict_keywords(word)
            if len(probable_word) > 0:
                tokenized_sentences[tokenized_sentences.index(word)] = probable_word

    taxonomy = Graph()
    results_lda, synonyms, keywords = process.process_text(" ".join(tokenized_sentences), )
    g = graph.load(os.path.join(model_path, "indicator_topics.graphml"))

    taxonomy.add_node("query", label=query_text)
    for (topic, weight) in results_lda[2]:
        taxonomy = graph.recursive_partition(g, taxonomy, topic)[0]
        for node in taxonomy.nodes():
            if node != "query":
                taxonomy.node[node]["label"] = topic_words[node]
            taxonomy.add_edge(topic, "query", weight=float(weight))

    # add indicators to vertices here. This isn't working.
    # traverse_graph(taxonomy, "query")

    graph.save("query", taxonomy)
    return taxonomy


if __name__ == "__main__":
    # test = query("health")
    pass
