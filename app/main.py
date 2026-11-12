from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import recover_jobs, router
from app.version import __version__


WEB_DIR = Path(__file__).resolve().parent / "web"


@asynccontextmanager
async def lifespan(application: FastAPI):
    recover_jobs()
    yield


app = FastAPI(title="AiClip", version=__version__, lifespan=lifespan)
app.include_router(router)
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
