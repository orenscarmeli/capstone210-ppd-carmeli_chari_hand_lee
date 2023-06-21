FROM python:3.10-slim as base

# install security updates
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
        curl \
    && rm -rf /var/lib/apt/lists/*

FROM base as build

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
        curl \
        build-essential \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install poetry
ENV POETRY_VERSION=1.3.2
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH /root/.local/bin:$PATH

# prepare venv
COPY poetry.lock pyproject.toml ./
RUN python -m venv --copies ./venv
RUN . /app/venv/bin/activate && poetry install --only main

ENV PATH /venv/bin:$PATH
# copy my src
COPY . ./

# prod
FROM base as prod

# update path
ENV PATH /app/venv/bin:$PATH

# copy
COPY --from=build /app/venv /app/venv/
COPY . ./

HEALTHCHECK --interval=5s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000:8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]