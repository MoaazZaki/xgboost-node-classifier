from pydantic import BaseModel


class Settings(BaseModel):
    """
    Class to hold service secrets and parameters.

    Attributes:
        api_key (str): Authentication key for secure connect.
        sentry_dsn (str): Sentry DSN to set up sentry logging tracking.
        model_path (str): Model load path.
        num_classes (int): Number of possible node classed/types.
        num_features (int): Feature vector dimension for each graph node to validate data.
        unknown_token (str): Label of missing
        redis_host (str): Redis host url for caching.
        cache_expiry (int): Threshold indicates the maximum time to send a rating for cached answer before deletion.
        opensearch_host (str): Opensearch host url for training data collection.
        opensearch_user (str): Opensearch user for authentication.
        opensearch_pass (str): Opensearch pass for authentication.
        opensearch_verify_certs (str):  Opensearch flag that indicates if we need to verify connection ssl certificates.
        opensearch_index (str): Opensearch index to store training data in.
        trainer_cron (str): Cron expression indicates how often the trainer watch if enough data is their to start.
        trainer_status_index (str): Trainer opensearch index to store training history.
        trainer_minimum_dataset_size (int): Training minimum new graphs to start new training iteration.
        app_host (str): App host, basically the ngnix client url that will be used from the trainer.
    """

    api_key: str
    sentry_dsn: str
    model_path: str
    num_classes: int
    num_features: int
    unknown_token: str
    redis_host: str
    cache_expiry: int
    opensearch_host: str
    opensearch_user: str
    opensearch_pass: str
    opensearch_verify_certs: bool
    opensearch_index: str
    trainer_cron: str
    trainer_status_index: str
    trainer_minimum_dataset_size: int
    app_host: str
