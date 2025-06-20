cd data/ssl4eo-s12/train/S2L1C

# Unzip all .zarr.zip files into .zarr folders
for f in *.zarr.zip; do
    unzip -q "$f" -d "${f%.zip}"
done
