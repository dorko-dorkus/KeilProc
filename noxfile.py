import nox
from pathlib import Path

ROOT = Path(__file__).parent

@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(
        "-r",
        "requirements.txt",
        "-c",
        "constraints.txt",
        "pytest",
    )
    session.run("pytest")

@nox.session
def smoke(session: nox.Session) -> None:
    """Exercise the CLI help to ensure the package imports."""
    session.install(
        "-r",
        "requirements.txt",
        "-c",
        "constraints.txt",
    )
    session.run("python", "-m", "kielproc.cli", "--help")
