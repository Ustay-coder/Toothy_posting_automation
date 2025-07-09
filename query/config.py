import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "USER": os.getenv("user"),
    "PASSWORD": os.getenv("password"),
    "HOST": os.getenv("host"),
    "PORT": os.getenv("port"),
    "DBNAME": os.getenv("dbname"),
}

# 조회할 스키마와 테이블명
SCHEMA = "toothpaste"
TABLE = "toothpastes" 
ITEM_NAME = "아조나치약"

# 치약 API 기본 URL
BASE_URL = "https://apis.kklim.io/toothy"
