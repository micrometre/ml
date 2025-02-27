FROM ubuntu:22.04 
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

COPY setup.sources.sh /setup.sources.sh
COPY setup.packages.sh /setup.packages.sh
COPY cpu.packages.txt /cpu.packages.txt
RUN /setup.sources.sh
RUN /setup.packages.sh /cpu.packages.txt


ARG PYTHON_VERSION=python3.11
ARG TENSORFLOW_PACKAGE=tf-nightly
COPY setup.python.sh /setup.python.sh
COPY cpu.requirements.txt /cpu.requirements.txt
RUN /setup.python.sh $PYTHON_VERSION /cpu.requirements.txt
RUN pip install --no-cache-dir ${TENSORFLOW_PACKAGE} 

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

COPY jupyter.requirements.txt /jupyter.requirements.txt
COPY setup.jupyter.sh /setup.jupyter.sh
RUN python3 -m pip install --no-cache-dir -r /jupyter.requirements.txt -U
RUN /setup.jupyter.sh

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

RUN useradd -m ubuntu
RUN echo "root:password" | chpasswd
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

USER root
RUN echo $HOME

WORKDIR /tf
EXPOSE 8888
EXPOSE 22

#CMD ["/usr/sbin/sshd", "-D"]

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
