#!/bin/bash

for file in $(find tests -type f -iname "*.jpg"); do
    echo "Processing: $file"
    python predict.py -l weights_resnet.pth "$file" -m RESNET
done