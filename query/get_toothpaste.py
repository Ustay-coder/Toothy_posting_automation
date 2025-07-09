import psycopg2
from config import DB_CONFIG, SCHEMA, TABLE, ITEM_NAME, BASE_URL
import requests
from utils import print_ingredients_by_csr_grade, save_ingredients_by_csr_grade_image

def get_toothpaste_by_name(item_name):
    try:
        connection = psycopg2.connect(
            user=DB_CONFIG["USER"],
            password=DB_CONFIG["PASSWORD"],
            host=DB_CONFIG["HOST"],
            port=DB_CONFIG["PORT"],
            dbname=DB_CONFIG["DBNAME"]
        )
        cursor = connection.cursor()
        query = f"SELECT * FROM {SCHEMA}.{TABLE} WHERE item_name = %s;"
        cursor.execute(query, (item_name,))
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return result
    except Exception as e:
        print(f'Error: {e}')
        return None

def get_toothpaste_id_by_name(item_name):
    try:
        connection = psycopg2.connect(
            user=DB_CONFIG["USER"],
            password=DB_CONFIG["PASSWORD"],
            host=DB_CONFIG["HOST"],
            port=DB_CONFIG["PORT"],
            dbname=DB_CONFIG["DBNAME"]
        )
        cursor = connection.cursor()
        query = f"SELECT id FROM {SCHEMA}.{TABLE} WHERE item_name = %s;"
        cursor.execute(query, (item_name,))
        result = cursor.fetchone()  # id는 하나만 반환될 것이므로 fetchone 사용
        cursor.close()
        connection.close()
        if result:
            return result[0]  # id 값만 반환
        else:
            return None
    except Exception as e:
        print(f'Error: {e}')
        return None

def get_toothpaste_data_by_id(toothpaste_id):
    try:
        url = f"{BASE_URL}/toothpastes/{toothpaste_id}"
        response = requests.get(url)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return response.json()  # JSON 데이터 반환
    except Exception as e:
        print(f'Error: {e}')
        return None


toothpaste_id = get_toothpaste_id_by_name(ITEM_NAME)
toothpaste_data = get_toothpaste_data_by_id(toothpaste_id)
if toothpaste_data:
    print_ingredients_by_csr_grade(toothpaste_data)
    save_ingredients_by_csr_grade_image(toothpaste_data)
else:
    print('치약 데이터를 불러오지 못했습니다.')