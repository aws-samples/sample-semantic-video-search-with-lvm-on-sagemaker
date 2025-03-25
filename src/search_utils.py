# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from collections import defaultdict, namedtuple


TimePoint = namedtuple('TimePoint', ['t', 'score', 'record_id'])
TimeCluster = namedtuple('TimeCluster', ['t_start', 't_end', 'score', 'top_point_searchid'])


def parse_opensearch_results(results):
    """Parse results from the opensearch call

    Args:
        results (dict): a dictionary from opensearch

    Returns:
        list: a list of TimePoint(time, score) tuples
    """
    output = defaultdict(list)
    for x in results['hits']['hits']:
        video_id = x['_source']['video_id']
        record_id = x['_id']
        point = TimePoint(t=float(x['_source']['timestamp']),
                          score=x['_score'], record_id=record_id)
        output[video_id].append(point)
    return output


def find_clusters(time_points, t_offset, as_dict=False):
    """Define clusters by grouping together neoghbours in time 
    descending it score

    Args:
        time_points (List[TimePoint(t, score)]): 
            a list of time points as tuples of time and score
        t_offset (float): 
            the maximum distance between the top point and an extra point we include in the cluster

    Returns:
        List[TimeCluster(t_start, t_end, score)]: 
            a list of found clusters sorted in a descending order
    """
    clusters = []
    clustered_points = []
    queue_points = sorted(time_points[:], key=lambda x: x.score, reverse=True)

    while len(queue_points):
        top_point = queue_points[0]
        # get the cluster
        cluster = [x for x in time_points if abs(x.t - top_point.t) <= t_offset]
        clustered_points += cluster
        clusters.append(
            TimeCluster(
                t_start=min(z.t for z in cluster),
                t_end=max(z.t for z in cluster),
                score=top_point.score,
                top_point_searchid=top_point.record_id
            )
        )

        queue_points = [x for x in queue_points if x not in clustered_points]

    if as_dict:
        clusters = [x._asdict() for x in clusters]
    return clusters
