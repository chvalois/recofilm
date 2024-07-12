from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import logging

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

def verify_mlflow():
    try:
        import mlflow
        logging.info("mlflow is available")
    except ModuleNotFoundError as e:
        logging.error(f"mlflow is not available: {str(e)}")
        raise

dag = DAG(
    'verify_mlflow_installation',
    default_args=default_args,
    description='Verify mlflow installation',
    schedule_interval='@once',
)

check_python_version = BashOperator(
    task_id='check_python_version',
    bash_command='python --version',
    dag=dag,
)

list_installed_packages = BashOperator(
    task_id='list_installed_packages',
    bash_command='pip list',
    dag=dag,
)

verify_mlflow_task = PythonOperator(
    task_id='verify_mlflow',
    python_callable=verify_mlflow,
    dag=dag,
)

check_python_version >> list_installed_packages >> verify_mlflow_task

