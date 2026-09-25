# syntax=docker/dockerfile:1
# Experimental Python-only image. Not a replacement for the FINN runtime image.
ARG PREPARED_IMAGE=xilinx/finn:deps-1d3775474e8454c3
FROM ${PREPARED_IMAGE} AS wheels
FROM finn-uv-spike:system
COPY uv /usr/local/bin/uv
ARG PROJECT_ROOT=/workspace/project
COPY pyproject.toml uv.lock ${PROJECT_ROOT}/
COPY local-packages.json /opt/finn/local-packages.json
ENV UV_PROJECT_ENVIRONMENT=/opt/finn/venv \
    UV_PYTHON=/usr/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=true
# Retain external wheels/build tools, never the source-provided distributions.
RUN --mount=from=wheels,source=/opt/finn/wheels,target=/wheel-input \
    python3 -c 'import json,shutil; from pathlib import Path; excluded=set(json.loads(Path("/opt/finn/local-packages.json").read_text())); target=Path("/opt/finn/wheels"); target.mkdir(parents=True); [shutil.copy2(p,target/p.name) for p in Path("/wheel-input").glob("*.whl") if p.name.split("-")[0].replace("_","-").lower() not in excluded]'
# No FINN or QONNX source directories are copied into this build context.
RUN uv sync --project ${PROJECT_ROOT} --frozen --offline --no-index \
    --find-links /opt/finn/wheels --no-install-local
ARG SPIKE_UID=1001
RUN useradd --create-home --uid ${SPIKE_UID} --shell /bin/bash agent && \
    chown -R agent:agent /opt/finn/venv
ENV PATH=/opt/finn/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
USER agent
WORKDIR /tmp
CMD ["python"]
