FROM python:3.13.1

RUN pip install uv

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" ./
RUN uv sync --locked

COPY "predict.py" "dv.bin" "eta=0.03max_depth=6min_child_weight=30.bin" ./

EXPOSE 9696

ENTRYPOINT [ "uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696" ]
