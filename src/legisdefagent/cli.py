import subprocess
from pathlib import Path

import typer

from legisdefagent.api.run_service import run_service
from legisdefagent.ingestion.embed_definitions import embed_definitions
from legisdefagent.ingestion.extract_definitions import extract_definitions
from legisdefagent.ingestion.populate_vectorstore import populate_vectorstore
from legisdefagent.settings import settings
from legisdefagent.core.utils import setup_logging


setup_logging()


app = typer.Typer(add_completion=False)
app.command(help="Extract definitions from local XML files or eXistDB collections.")(extract_definitions)
app.command(help="Compute embeddings for extracted definitions.")(embed_definitions)
app.command(help="Populate the vector store with definition embeddings and metadata.")(populate_vectorstore)
app.command(help="Start the backend agent service.")(run_service)


@app.command("run-app", help="Start the Streamlit frontend application.")
def run_app():
    script_path = Path(__file__).resolve().parent / "frontend" / "streamlit_app.py"
    subprocess.run(["streamlit", "run", str(script_path), f"--server.port={settings.ui.port}"])


if __name__ == "__main__":
    app()
