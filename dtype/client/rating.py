from pydantic import BaseModel


class Rating(BaseModel):
    """
    Class to hold rating information.

    Attributes:
        id (str): Rating track uuid.
        label (str): Correct answer.
    """

    id: str
    label: str
