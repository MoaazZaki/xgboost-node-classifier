from typing import List

from pydantic import BaseModel

from dtype.client.rating import Rating
from dtype.model.graph import Graph


class PredictBody(BaseModel):
    """
    Class to hold predict endpoint request body.

    Attributes:
        graphs (List[Graph]): List of input adjacency list based graphs to perform prediction on.
        track (:obj:`bool`, optional): Flag to enable caching and returning of track uuids of each prediction. Defaults
            to False.
    """

    graphs: List[Graph]
    track: bool = False


class RatingBody(BaseModel):
    """
    Class to hold rate endpoint request body.

    Attributes:
        ratings (List[Rating]): List of input ratings (id & label) to provide feedback and store incorrect predictions
            for re-training.
    """

    ratings: List[Rating]
