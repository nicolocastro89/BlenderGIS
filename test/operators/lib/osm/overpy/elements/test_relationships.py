import unittest
from operators.lib.osm.overpy.OSMLibrary import OSMLibrary
from Mock_data import json_nodes,json_ways,json_relationships


class TestMultipolygonRelationships(unittest.TestCase):

    def one_outer_one_inner(self):
        elements = []
        elements.extend(json_nodes)
        elements.extend(json_ways)
        elements.append(json_relationships['one_outer_one_inner'])
        library = OSMLibrary.from_json({
            'elements': elements
        })
