# Project Context

This file contains important context and notes for Claude Code.

## Testing Commands
- Run tests: `poetry run test` or `poetry run tests`
- Run specific test: `poetry run pytest tests/path/to/test.py`
- Run with coverage: `poetry run pytest --cov`
- Run unit tests only: `poetry run pytest -m unit`
- Run integration tests only: `poetry run pytest -m integration`
- Run without slow tests: `poetry run pytest -m "not slow"`

## Linting Commands
- To be determined after setup

## Project Structure
- Tests are located in `tests/` directory
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Test configuration in `pyproject.toml`
- Shared fixtures in `tests/conftest.py`

## Package Management
- Using Poetry for dependency management
- Install dependencies: `poetry install`
- Add new dependency: `poetry add <package>`
- Add dev dependency: `poetry add --group dev <package>`