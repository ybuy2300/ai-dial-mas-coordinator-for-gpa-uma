import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import MASCoordinator
from task.logging_config import setup_logging, get_logger

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
UMS_AGENT_ENDPOINT = os.getenv('UMS_AGENT_ENDPOINT', "http://localhost:8042")
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

setup_logging(log_level=LOG_LEVEL)
logger = get_logger(__name__)


class MASCoordinatorApplication(ChatCompletion):

    async def chat_completion(self, request: Request, response: Response) -> None:
        # 1. Create single choice with context manager
        # 2. Create MASCoordinator and handle request

        try:
            with response.create_single_choice() as choice:

                await MASCoordinator(
                    endpoint=DIAL_ENDPOINT,
                    deployment_name=DEPLOYMENT_NAME,
                    ums_agent_endpoint=UMS_AGENT_ENDPOINT
                ).handle_request(
                    choice=choice,
                    request=request,
                )

                logger.info("Successfully completed chat request")

        except Exception as e:
            logger.error(
                f"Error processing chat completion: {str(e)}",
                exc_info=True
            )
            raise



# 1. Create DIALApp
# 2. Create MASCoordinatorApplication
# 3. Add to created DIALApp chat_completion with:
#       - deployment_name="mas-coordinator"
#       - impl=agent_app
# 4. Run it with uvicorn: `uvicorn.run({CREATED_DIAL_APP}, port=8055, host="0.0.0.0")`
logger.info("Creating DIAL application")
app: DIALApp = DIALApp()
agent_app = MASCoordinatorApplication()
app.add_chat_completion(deployment_name="mas-coordinator", impl=agent_app)
logger.info("DIAL application initialized successfully")

if __name__ == "__main__":
    logger.info("Starting DIAL application with Uvicorn")
    uvicorn.run(app, port=8055, host="0.0.0.0")   