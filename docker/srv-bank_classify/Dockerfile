FROM rocker/r-base
MAINTAINER "Christoph Fabianek" christoph@ownyourdata.eu

RUN R -e "install.packages(c('jsonlite', 'NLP', 'tm', 'naivebayes'), repos='http://cran.r-project.org')" \
	&& mkdir -p /srv-bank_classify

USER docker

COPY script/run.sh /bin/run.sh
COPY script/*.R /srv-bank_classify

CMD ["/bin/bash"]