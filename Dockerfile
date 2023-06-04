FROM python:3.9

WORKDIR /app
COPY app .
COPY req req
RUN pip install -r req
RUN mkdir -p ~/.streamlit


COPY config.toml ~/.streamlit/config.toml

CMD ["streamlit", "run" , "bot.py","--server.port","80"]
