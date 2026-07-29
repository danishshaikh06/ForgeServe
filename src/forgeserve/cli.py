"""Console script for forgeserve."""

import typer
from rich.console import Console

from forgeserve import utils

app = typer.Typer()
console = Console()


@app.command()
def main() -> None:
    """Console script for forgeserve."""
    console.print("Replace this message by putting your code into forgeserve.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()

@app.command()
def ForgeServe() -> None:
    """Run the ForgeServe application."""
    print("Coming soon! This command will start the ForgeServe application.")

@app.command()
def AvailableCommands() -> None:
    """Display available commands."""
    print("Coming soon! This command will display available commands.")

@app.command()
def serve() -> None:
    """Run the Serve command."""
    print("Coming soon!.")

@app.command()
def version() -> None:
    """Give ForgeServe Version."""
    print("ForgeServe v0.1.0")

@app.command()
def benchmark() -> None:
    """Gives benchmark."""
    print("Coming Soon!.")


if __name__ == "__main__":
    app()
