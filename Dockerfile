FROM python:3.12-slim

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock /app/

# Set the working directory to /app
WORKDIR /app

# Install dependencies (using cache)
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# Copy the current directory contents into the container at /app
COPY main.py /app/main.py

# Expose port 8000
EXPOSE 8000

# Run the app
ENTRYPOINT ["uv", "run", "--no-dev", "uvicorn", "main:app", "--host", "0.0.0.0"]
