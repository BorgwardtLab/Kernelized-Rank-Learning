#!/bin/bash

python dataset.py URLs_to_GDSC_datasets.txt
bash prepare_KBMTL.sh

for analysis in "KEEPK" "SAMPLE"
do
	cp test/${analysis}_config.yaml config.yaml
	python prepare.py config.yaml
	python exp.py config.yaml
	matlab -nodisplay -nosplash -nodesktop -r "run_KBMTL"
	python results.py config.yaml > ${analysis}_results.txt
done

