import sqlite3
import pandas as pd


def get_potential_ddi(drug_name):
    conn = sqlite3.connect('./dataset/drug_interactions.db')

    query = f"""SELECT 
    CASE 
        WHEN Drug_1 = 'abciximab' THEN Drug_2 
        ELSE Drug_1 
    END AS Other_Drug,
    ADE
FROM 
    drug_interactions
WHERE 
    Drug_1 = '{drug_name}' OR Drug_2 = '{drug_name}';"""

    result = pd.read_sql_query(query, conn).values.tolist()
    conn.close()
    return result
