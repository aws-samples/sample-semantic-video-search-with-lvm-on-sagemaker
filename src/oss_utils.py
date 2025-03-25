# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import boto3
from requests_aws4auth import AWS4Auth

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy import helpers as opensearch_helpers


def oss_ingest_batch(index_name, batch_embeddings, batch_timestamps, video_name, oss_client):
    '''
    Takes one batch per iteration of given size (e.g. 32) and send documents as bulk to opensearch index
    '''

    actions = []
    for x, t in zip(batch_embeddings.tolist(), batch_timestamps.tolist()):
        actions.append({
            "_op_type": "index",
            "_index": index_name,
            "_source": {
                "frame_vector": x,
                "timestamp": str(t),
                "video_id": video_name,
            },
        })

    opensearch_helpers.bulk(oss_client, actions)
    return len(batch_embeddings)


def get_oss_client(collection_id, region):
    service = 'aoss'    # Amazon OpenSearch Serverless
    credentials = boto3.Session().get_credentials()
    host = f"{collection_id}.{region}.aoss.amazonaws.com"
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

    return OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )


def get_oss_query_for(vec, search_size, k, search_name=None):
    # TODO: with search_name we don't get any results which is strange
    if search_name:
        return {
            "size": search_size,
            "query": {
                "bool": {
                    "filter": {
                        "bool": {
                            "must": [
                                {
                                    "term": {
                                        "video_id": search_name
                                    }
                                }
                            ]
                        }
                    },
                    "must": [
                        {
                            "knn": {
                                "frame_vector": {
                                    "vector": vec,
                                    "k": k
                                }
                            }
                        }
                    ]
                }
            }
        }

    else:
        return {
            "size": search_size,
            "query": {
                "knn": {
                    "frame_vector": {
                        "vector": vec,
                        "k": k
                    }
                }
            }
        }
