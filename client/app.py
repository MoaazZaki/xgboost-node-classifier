import logging
import uuid
from datetime import datetime
from typing import Union

from fastapi import FastAPI, Header, Response, status
from fastapi.middleware.cors import CORSMiddleware

from client.manager import Manager
from dtype.client.body import PredictBody, RatingBody
from model.xgboost import GraphXGB

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
manager: Union[Manager, None] = None
model: Union[GraphXGB, None] = None


@app.on_event("startup")
def startup_event() -> None:
    """
    App on start function to set up manager and load ML model.
    """
    global manager, model
    manager = Manager.setup()

    model = GraphXGB(
        path=manager.settings.model_path,
        num_features=manager.settings.num_features,
        num_classes=manager.settings.num_classes,
        unknown_token=manager.settings.unknown_token,
    )


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
@app.get("/health", response_class=Response)
@app.get("/", response_class=Response)
async def live_and_ready(response: Response) -> None:
    """
    Health endpoint for the service.
    Args:
        response (Response): Fastapi response object that holds the status code.
    """
    response.status_code = status.HTTP_204_NO_CONTENT


@app.post("/predict")
@app.post("/predict/")
async def predict(body: PredictBody, response: Response, api_key: str = Header("")) -> dict:
    """
    Predict endpoint for the app.
    Args:
        body (PredictBody): Request body consists of inputs graphs and flag to return track IDs,
            see dtype/client/body.py for model details.
        response (Response): Fastapi response object that holds the status code.
        api_key (:obj:`str`, optional): Value of Api-key header Defaults to "".

    Returns:
        dict: Response body, the fields (results & track) are returned if the processing is successful, error field
            is returned otherwise.
    """
    if api_key != manager.settings.api_key:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Incorrect or empty Api-Key header."}

    try:
        response_body = {"results": await model(body.graphs)}

        if body.track:
            track_ids = [str(uuid.uuid4()) for _ in range(len(body.graphs))]
            await manager.cache_results(ids=track_ids, body=body, results=response_body["results"])
            response_body["track"] = track_ids

        response.status_code = status.HTTP_200_OK
        return response_body
    except Exception as e:
        Manager.log(
            message=f"Failed to perform prediction with error {str(e)} for input {body.graphs}.",
            level=logging.ERROR,
            file_path=__file__,
        )

        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": f"Failed to perform prediction with error {str(e)}."}


@app.post("/rate")
@app.post("/rate/")
async def rate(body: RatingBody, response: Response, api_key: str = Header("")):
    """
    Rate endpoint for the app.
    Args:
        body (PredictBody): Request body consists of ratings, each rating has track id and the actual label.
            see dtype/client/body.py for model details.
        response (Response): Fastapi response object that holds the status code.
        api_key (:obj:`str`, optional): Value of Api-key header. Defaults to "".

    Returns:
        dict: Response body, the fields (results & track) are returned if the processing is successful, error field is
            returned otherwise.
    """

    if api_key != manager.settings.api_key:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Incorrect or empty Api-Key header."}

    try:
        all_wrong_answers = []
        for rating in body.ratings:
            result = await manager.get_from_cache(id=rating.id)
            if result["output"] != rating.label:
                result["output"] = rating.label
                result['added_timestamp'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')

                all_wrong_answers.append(result)

        await manager.store_wrong_answers(answers=all_wrong_answers)
        response.status_code = status.HTTP_200_OK
        return {"results": ""}
    except Exception as e:
        error_msg = f"Failed to perform rating with error {str(e)}."
        Manager.log(message=error_msg, level=logging.ERROR, file_path=__file__)

        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": error_msg}


@app.post("/upgrade")
@app.post("/upgrade/")
async def upgrade(response: Response, api_key: str = Header("")):
    """
    Upgrade endpoint to re-load the new trained model.
    Args:
        response (Response): Fastapi response object that holds the status code.
        api_key (:obj:`str`, optional): Value of Api-key header. Defaults to "".

    Returns:
        dict: Response body, the fields (results & track) are returned if the processing is successful, error field is
            returned otherwise.
    """
    if api_key != manager.settings.api_key:
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {"error": "Incorrect or empty Api-Key header."}

    global model
    try:
        model = GraphXGB(
            path=manager.settings.model_path,
            num_features=manager.settings.num_features,
            num_classes=manager.settings.num_classes,
            unknown_token=manager.settings.unknown_token,
        )
    except Exception as e:
        error_msg = f"Failed to upgrade with error {str(e)}."
        Manager.log(message=error_msg, level=logging.ERROR, file_path=__file__)

        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": error_msg}

    response.status_code = status.HTTP_200_OK
