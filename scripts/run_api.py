import uvicorn

from traffic_law_v2.api import app
from traffic_law_v2.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    main()

