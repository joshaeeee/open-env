"""FastAPI wiring for OpenER."""

from __future__ import annotations

from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import create_app

try:
    from ..models import ERAction, ERObservation
    from .environment import ERTriageEnvironment
except ImportError:  # pragma: no cover - direct source execution fallback
    from models import ERAction, ERObservation  # type: ignore
    from server.environment import ERTriageEnvironment  # type: ignore


app = create_app(ERTriageEnvironment, ERAction, ERObservation, env_name="open_er")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/admin", methods=["GET", "HEAD"], include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":  # pragma: no cover
    main()
