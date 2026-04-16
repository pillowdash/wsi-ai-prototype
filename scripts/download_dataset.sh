

aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON16/images/ | grep tumor_ | head -n 20

# Download the normal slide
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/images/normal_019.tif   ~/wsi-ai-prototype/data/raw/camelyon16/images/

# Download the tumor slide
aws s3 cp --no-sign-request s3://camelyon-dataset/CAMELYON16/images/tumor_001.tif ~/wsi-ai-prototype/data/raw/camelyon16/images/

# annotation files
aws s3 cp --no-sign-request \
  s3://camelyon-dataset/CAMELYON16/annotations/tumor_001.xml ~/wsi-ai-prototype/data/raw/camelyon16/annotations/


# Save the listings
aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON16/images/ > data/raw/camelyon16/metadata/images_list.txt

aws s3 ls --no-sign-request s3://camelyon-dataset/CAMELYON16/annotations/ > data/raw/camelyon16/metadata/annotations_list.txt
