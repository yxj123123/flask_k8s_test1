FROM python
RUN mkdir -p /Federated \
    && mkdir -p /Federated/templates \
    && mkdir -p /Federated/static
COPY ./templates/index.html /Federated/templates
COPY main.py /Federated/
WORKDIR /Federated
RUN pip install -r requirements.txt
EXPOSE 5000
RUN /bin/bash -c 'echo init ok'
CMD ["python", "main.py"]
