FROM python:3.11-slim-bookworm AS compiled

ARG MPI_UID=20001
ARG MPI_GID=4322
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenmpi-dev openmpi-bin openssh-server openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${MPI_GID} mpiuser \
    && useradd -m -u ${MPI_UID} -g ${MPI_GID} -s /bin/bash mpiuser \
    && mkdir -p /run/sshd

WORKDIR /opt/easygraph
RUN python -m pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN python -m pip install --no-cache-dir setuptools==75.8.0 wheel Cython pybind11==3.0.1 \
    && CMAKE_BUILD_PARALLEL_LEVEL=2 EASYGRAPH_ENABLE_DISTRIBUTED=ON \
    python -m pip install --no-cache-dir --no-build-isolation . \
    && rm -rf build

FROM compiled AS runtime
RUN python -m pip check && python -c 'import easygraph.distributed, cpp_easygraph_distributed'
COPY deploy/k8s /opt/easygraph/deploy/k8s
COPY deploy/k8s/ssh-client.conf /etc/ssh/ssh_config.d/easygraph.conf
COPY deploy/k8s/sshd_config /home/mpiuser/.sshd_config
COPY deploy/k8s/start-worker.sh /usr/local/bin/start-easygraph-worker
RUN chown mpiuser:mpiuser /home/mpiuser/.sshd_config
USER mpiuser
ENV PATH=/usr/local/bin:/usr/bin:/bin
WORKDIR /home/mpiuser
EXPOSE 2222
CMD ["bash", "/usr/local/bin/start-easygraph-worker"]
