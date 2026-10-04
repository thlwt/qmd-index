FROM python:3.13-slim
WORKDIR /app

COPY webui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qmd.py cli.js package.json qmd.yml qmd.example.yml qmd_mcp_server.py ./
COPY translator.py ragflow_bridge.py ./
COPY webui/ webui/

EXPOSE 8090
CMD ["python", "webui/server.py"]
