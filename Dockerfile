FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/eeg_report_multiagent_v1

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

ARG INSTALL_EVAL_DEPS=0
ARG INSTALL_BERTSCORE_DEPS=0
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$INSTALL_EVAL_DEPS" = "1" ] && [ "$INSTALL_BERTSCORE_DEPS" = "1" ]; then \
      pip install --no-cache-dir -e ".[dev,eval,bertscore]"; \
    elif [ "$INSTALL_EVAL_DEPS" = "1" ]; then \
      pip install --no-cache-dir -e ".[dev,eval]"; \
    else \
      pip install --no-cache-dir -e ".[dev]"; \
    fi

CMD ["bash"]
