from typing import List

from pydantic import BaseModel

from dtype.model.edge import Edge


class Graph(BaseModel):
    """
    Class to hold adjacency list based graph information.

    Attributes:
        List[Edge] (List[Edge]): Edges list of the graph.
    """

    edges: List[Edge]
