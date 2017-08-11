#!/bin/bash

python dataset.py URLs_to_GDSC_datasets.txt
for analysis in "KEEPK" "SAMPLE" #"FULL"
do
	cp test/${analysis}_config.yaml config.yaml
	python prepare.py config.yaml
	python exp.py config.yaml
	bash prepare_KBMTL.sh
	matlab -nodisplay -nosplash -nodesktop -r "run_KBMTL"
	python results.py config.yaml > ${analysis}_results.txt
done
jupyter notebook plots.ipynb

