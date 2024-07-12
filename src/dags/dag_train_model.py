from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from src.models.train_model_svd import train_svd_model
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow/src/models')

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

my_dag = DAG(
    dag_id='train_model',
    description='train SVD model',
    tags=['train_model'],
    schedule_interval='0 2 * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    }
)

task_1 = PythonOperator(
    task_id='train_model',
    python_callable=train_svd_model,
    dag=my_dag
)
