#!/usr/bin/env bash
# cd scratch place

# Download perception from Google Drive
filename='dataset.zip'
fileid='106cdnUk7lxxvj_7Dd_nKG7R4bWn3w7Dk'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

# Unzip
unzip -q ${filename}
rm ${filename}