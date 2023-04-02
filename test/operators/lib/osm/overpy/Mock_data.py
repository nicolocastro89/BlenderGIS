json_nodes = [
    {
    'type': 'node','id': 1,'lat': 44.4025089,'lon': 8.9456073
    },
    {
    'type': 'node','id': 2,'lat': 44.4025089,'lon': 8.9456073,'tags': {}
    },
    {
    'type': 'node','id': 3,'lat': 44.4025089,'lon': 8.9456073
    },
    {
    'type': 'node','id': 4,'lat': 44.4025089,'lon': 8.9456073
    },
    {
    'type': 'node','id': 5,'lat': 44.4025089,'lon': 8.9456073
    },
    {
    'type': 'node','id': 6,'lat': 44.4025089,'lon': 8.9456073
    },
    {
    'type': 'node','id': 7,'lat': 44.4025089,'lon': 8.9456073
    },
    {
    'type': 'node','id': 8,'lat': 44.4025089,'lon': 8.9456073
    },
    {
    'type': 'node','id': 9,'lat': 44.4025089,'lon': 8.9456073
    },
]

json_ways = [
    {'type': 'way','id': 100,
    'nodes': [1,2,3,4,5,1],
    'tags': {}
    },
    {'type': 'way','id': 200,
    'nodes': [6,7,8,9,6],
    'tags': {}
    },
]

json_relationships = [
    'one_outer_one_inner': {
        'type': 'relation', 'id': 1000,
        "tags": {"type": "multipolygon"}
        'members': [
                {'type': 'way','ref': 100,'role': 'outer'
                },
                {'type': 'way','ref': 200,'role': 'inner'
                }
            ],
    }
]