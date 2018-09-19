# Clinical trials prediction

This directory includes scripts to reproduce our preliminary analysis of predicting patient responses in docetaxel and bortezomib clinical trials by training on cancer cell lines from GDSC (see Section 3.4.2 in our paper, [*Bioinformatics*, **34**(16), 2808–2816](http://doi.org/10.1093/bioinformatics/bty132)). We used the supplementary scripts and datasets from [Geeleher *et al.* (2014), *Genome Biology*, **15**, R47](https://doi.org/10.1186/gb-2014-15-3-r47) as our starting point. Since Geeleher *et al.* (2014) developed their approach using GDSC release 2 (KRL was developed with GDSC release 6), we spiked-in the GDSC release 6 data into their pipeline. These spiked-in scripts are located here.

## Contents

The python script **create_data.py** generates CSV files with GDSC release 6 gene expression, docetaxel IC50s, and bortezomib IC50s. The files in the **data** directory contain the clinical trials data from the Geeleher's *et al.* (2014) supplement. The two R scripts are based on Geeleher's *et al.* (2014) code, adjusted for spiking-in the GDSC release 6 data. The constant variable **R2R6CCL** in those R scripts has the following meaning. When *FALSE*, the pipeline will use all the GDSC release 6 cell lines (833 and 391 cell lines for docetaxel and bortezomib, respectively). When TRUE, it will take the intersection of cell lines present in both GDSC release 2 and GDSC release 6 (394 and 253 cell lines for docetaxel and bortezomib, respectively). The **scripts** folder is from Geeleher's *et al.* (2014) supplement with no edits.

## Usage

Create CSV files with GDSC release 6 gene expression, docetaxel IC50s, and bortezomib IC50s:

    python create_data.py

Run docetaxel prediction:

    Rscript docetaxelBreastCancer_GDSCr6.R

Run bortezomib prediction:

    Rscript bortezomib_GDSCr6.R


## Additional dependancies

The above presumes that you have already downloaded the GDSC release 6 datasets:

    cd ..
    python dataset.py URLs_to_GDSC_datasets.txt

I tested the R scripts with R version 3.5.1 (2018-07-02) and the following libraries installed:

    sudo apt install r-base r-base-dev
    sudo apt install libxml2 libxml2-dev libgsl-dev libcurl4-openssl-dev
    R
    source("https://bioconductor.org/biocLite.R")
    biocLite()
    biocLite("sva")
    biocLite("GEOquery")
    biocLite("hgu133a.db")
    biocLite("hgu133b.db")
    install.packages("gsl")
    install.packages("ridge")
    install.packages("ROCR")
    install.packages("curl")
    install.packages("car")

