from pydantic import BaseModel


class Edge(BaseModel):
    """
    Class to hold graph edge information.

    Attributes:
        origin_id (str): Source node ID.
        origin_type (str): Source node type.
        destination_id (str): Destination node ID
        destination_type (str): Destination node type.
    """

    origin_id: str
    origin_type: str
    destination_id: str
    destination_type: str
