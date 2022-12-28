#!/usr/bin/env bash
# cd scratch place
mkdir -p checkpoint
cd checkpoint/

# Download perception from Google Drive
filename='perception.zip'
fileid='18EKIOWNdMObRjtcLaMIVkI8qh8MvFWaT'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

# Unzip
unzip -q ${filename}
rm ${filename}

# Download PLATO from Google Drive
filename='PLATO.zip'
fileid='1mucFuRzlwh-7ZD7Z-ig3wH9nX_YkKl02'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

# Unzip
unzip -q ${filename}
rm ${filename}

# Download XPL from Google Drive
filename='XPL.zip'
fileid='1Lc7tT3UfePtDPzyJ4_U8TQ_y2zwmMt4o'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt
# Unzip
unzip -q ${filename}
rm ${filename}