import logging
import uuid
from time import time, sleep

import requests
from croniter import croniter

from client.manager import Manager
from trainer.learner import Learner
from trainer.loader import Loader

SECOND = 1


def start():
    manager = Manager.setup()

    iterator = croniter(manager.settings.trainer_cron)
    next_run = iterator.get_next(float)
    while True:
        now = time()
        if next_run <= now:
            run_uuid = uuid.uuid4()
            manager.log(
                message=f"Starting a try to train with track ID:{run_uuid}.",
                level=logging.DEBUG,
                file_path=__file__
            )

            if Loader.should_start(manager=manager):
                manager.log(
                    message=f"New training session with track ID:{run_uuid} has started.",
                    level=logging.INFO,
                    file_path=__file__
                )

                raw_data = Loader.get(manager=manager)

                learner = Learner(manager=manager)
                features, labels = learner.prepare(raw_data=raw_data)
                del raw_data

                manager.log(
                    message=f"Preprocessing of training session with track ID:{run_uuid} is done, starting training.",
                    level=logging.DEBUG,
                    file_path=__file__
                )

                learner.start(features=features, labels=labels)
                del features
                del labels

                manager.log(
                    message=f"Fitting the mode is done in session with track ID:{run_uuid} .",
                    level=logging.INFO,
                    file_path=__file__
                )

                learner.save(manager=manager)
                learner.clean()

                requests.post(
                    headers={"Api-key": manager.settings.api_key},
                    url=f'{manager.settings.app_host}/upgrade'
                ).raise_for_status()

                manager.log(
                    message=f"Training session with track ID:{run_uuid} is done successfully.",
                    level=logging.INFO,
                    file_path=__file__
                )
            else:
                manager.log(
                    message=f"No training session will start for track ID:{run_uuid}, not enough data.",
                    level=logging.DEBUG,
                    file_path=__file__
                )

            next_run = iterator.get_next(float)
        sleep(SECOND)


if __name__ == '__main__':
    start()
