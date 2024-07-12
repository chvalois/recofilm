from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import sys

# Add the tests directory to the Python path
sys.path.append('/opt/airflow')

def run_tests():
    try:
        result = subprocess.run(["pytest", "tests/test_models/test_predict_hybrid.py"], check=True, 
capture_output=True, text=True)
        print("Test Output:\n", result.stdout)
        print("Pytest Error Output:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error Output:\n", e.stderr)
        print("Pytest Output (if any):\n", e.stdout)
        raise


default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2024, 6, 20),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

my_dag = DAG(
    dag_id='hybrid_reco_tests', 
    description='Run hybrid tests', 
    tags=['hybrid_reco_tests'],
    schedule_interval='0 0-23/4 * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    }
)

task_1 = PythonOperator(
    task_id='run_hybrid_tests', 
    python_callable=run_tests, 
    dag=my_dag,  
)



