from dotenv import load_dotenv, find_dotenv
import torch
import certifi
import os

# Load environment variables
load_dotenv(find_dotenv())

# Set defaults
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", certifi.where())
os.environ["SSL_CERT_FILE"] = SSL_CERT_FILE  # also write to os.environ so requests can find it
DATA_DIR = os.getenv("DATA_DIR", "data")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
