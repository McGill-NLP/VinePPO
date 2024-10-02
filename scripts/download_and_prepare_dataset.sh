#!/bin/bash

# Download the zip file
wget -O wandb_export_root.zip https://github.com/McGill-NLP/vineppo/releases/download/latest/wandb_export_root.zip

# Create the data directory
mkdir -p data

# Extract wandb_export_root.zip
unzip wandb_export_root.zip -d zip_root

# Extract the archives inside datasets directory into data
cd zip_root/wandb_export_root/datasets

for archive in *.zip; do
  # Filename template: data-ds_name.zip
  # Extract the ds_name
  ds_name="${archive#data-}"
  ds_name="${ds_name%.zip}"
  mkdir -p "../../../data/$ds_name"

  unzip "$archive" -d "../../../data/$ds_name"
  cp -r "../../../data/$ds_name"/* "../../../data/"
  rm -rf "../../../data/$ds_name"
done

echo "Extraction completed!"

# Remove the zip_root directory
cd ../../..
rm -rf zip_root

# Remove the zip file
rm wandb_export_root.zip