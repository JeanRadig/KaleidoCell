# Comparison against geneNMF

We provide here the resources utilised to run geneNMF on GBmap for runtime comparison against kaleidoCell. Please refer to the details in the paper.

- GBmap download: https://cellxgene.cziscience.com/collections/999f2a15-3d7e-440b-96ae-2c806799c08c

- genenmf docker: publicly available at `hdsu/genenmf_env`

You can also build the docker from scratch using the Dockerfile present in comparison/genenmf_docker.

To run the experiment, first you will need to downsample GBmap to multiple fraction of the total amount of cells. Then, run `comparison/genenmf_runtime_script.R` in the docker. 

