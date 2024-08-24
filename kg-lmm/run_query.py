#!/usr/bin/env python3

import os

from langchain.chains import GraphSparqlQAChain

from langchain_community.graphs import RdfGraph
# from langchain_community.llms.ollama import Ollama

from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI

import networkx as nx
import matplotlib.pyplot as plt
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph

RDF_FILEPATH = ""
QUERY = "What are the parameter names of the Authentication Service?"
# MODEL = "orca-mini"


def read_rdf(filename):
    graph = RdfGraph(source_file=filename)
    graph.load_schema()
    return graph


def query(query, llm, graph):
    chain = GraphSparqlQAChain.from_llm(llm, graph=graph, verbose=True)
    output = chain.invoke({"query": query})
    return output[chain.output_key]


def print_graph(graph):
    G = rdflib_to_networkx_multidigraph(graph.graph)
    pos = nx.spring_layout(G, scale=2)
    edge_labels = nx.get_edge_attributes(G, 'r')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw(G, with_labels=True)
    plt.show()


def main():
    graph = read_rdf(RDF_FILEPATH)
    # print_graph(graph)

    # llm = Ollama(model=MODEL, temperature=0)
    llm = AzureChatOpenAI(deployment_name="", temperature=0)
    print(f"Running query: '{QUERY}'")
    output = query(QUERY, llm, graph)
    print(f"Results: '{output}'")


if __name__ == "__main__":
    os.environ["AZURE_OPENAI_ENDPOINT"] = ""
    os.environ["AZURE_OPENAI_API_KEY"] = ""
    os.environ["OPENAI_API_VERSION"] = ""
    main()
