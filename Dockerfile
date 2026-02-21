FROM python:3.11-slim

# working directory inside container
WORKDIR /app

# copy project files
COPY . .

# create virtual environment
RUN python -m venv /app/.venv

# activate venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# create mount points
VOLUME /app/dati
VOLUME /app/plots
VOLUME /app/performances

# run the project
ENTRYPOINT ["python","main.py"]
