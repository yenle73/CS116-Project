#!/bin/bash

# Cài đặt Python 3.10.12 (hoặc phiên bản tương tự)
sudo apt update
sudo apt install -y python3.10 python3.10-venv

# Tạo môi trường ảo
python3.10 -m venv venv
source venv/bin/activate

# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt