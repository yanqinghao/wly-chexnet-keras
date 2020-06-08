ARG PYTHON_VERSION

ARG NAME_SPACE

ARG DOCKER_BASE

FROM registry-vpc.cn-shanghai.aliyuncs.com/${NAME_SPACE}/${DOCKER_BASE}:${PYTHON_VERSION} as builder

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pyarmor

WORKDIR /build

COPY . /build

RUN bash tools/compress.sh

ARG PYTHON_VERSION

ARG NAME_SPACE

ARG DOCKER_BASE

ARG REQUIREMENTS_FILE

FROM registry-vpc.cn-shanghai.aliyuncs.com/${NAME_SPACE}/${DOCKER_BASE}:${PYTHON_VERSION}

WORKDIR /wly

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

COPY --from=builder /build /wly

RUN pip install -r REQUIREMENTS_FILE

ENTRYPOINT [ "/sbin/my_init", "--" ]

CMD [ "bash" ]
