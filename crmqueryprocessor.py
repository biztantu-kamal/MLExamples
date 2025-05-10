from openai import OpenAI
import psycopg2
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import re
from dotenv import load_dotenv
import os
import time
import logging

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)

class CRMQueryProcessor:
    def __init__(self):
        # Load configuration from environment variables
        db_host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        
        # Log database configuration status (without sensitive data)
        logger.info(f"Database configuration status:")
        logger.info(f"DB_HOST: {'Set' if db_host else 'Missing'}")
        logger.info(f"DB_NAME: {'Set' if db_name else 'Missing'}")
        logger.info(f"DB_USER: {'Set' if db_user else 'Missing'}")
        logger.info(f"DB_PASSWORD: {'Set' if db_password else 'Missing'}")
            
        self.db_config = {
            "host": db_host,
            "database": db_name,
            "user": db_user,
            "password": db_password,
            "port": os.getenv("DB_PORT", "5432")
        }
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 1

        self.schema = """
        Tables and their relationships:
        
        ordertab (order_id, client_id, customer_id, lead_id, status, type, created, updated, deleted)
        - Links to customer through customer_id
        - Links to lead through lead_id
        
        customer (customer_id, client_id, name, email, phone, income_group, marital_status, source)
        - Referenced by ordertab
        
        lead (lead_id, client_id, user_id, name, email, phone, status, source, income)
        - Referenced by ordertab
        - Has followups in leadfollowup
        
        product (product_id, client_id, name, price, sale_commission, status, type)
        - Used in order_product_mapping and leadproduct
        
        order_product_mapping (id, order_id, product_id, product_quantity, rate)
        - Links orders to products
        """

    def clean_sql(self, sql: str) -> str:
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*$', '', sql)
        return sql.strip()

    def enhance_query(self, natural_query: str, attempt: int = 1) -> str:
        prompt = f"""
        Given this database schema:
        {self.schema}
        
        Important notes:
        1. Sample data is from Jan-Mar 2024, use this date range for "current" queries
        2. Use proper JOIN conditions and handle NULL values
        3. For revenue calculations, use product_quantity * rate
        4. Always check deleted = false or IS NULL where applicable
        5. Use CAST for decimal calculations
        
        This is attempt {attempt} of {self.MAX_RETRIES}.
        If previous attempts failed, generate a different variation of the query.
        
        Convert this business question to PostgreSQL (return ONLY the SQL): {natural_query}
        
        Important schema and PostgreSQL notes:
        - User table is named 'bizuser' (not 'users')
        - User fields are first_name, last_name (not name)
        - Lead status values are: NEW, IN_PROGRESS, QUALIFIED, CONVERTED
        - Order status values are: NEW, IN_PROGRESS, COMPLETED, DELIVERED
        - For decimal calculations, use CAST(value AS numeric(10,2))
        - For aggregations with decimals, use CAST(SUM/AVG as numeric(10,2))
        """
        
        if "this month" in natural_query.lower():
            prompt += "\nNote: Use January 2024 as 'this month' for the sample data."
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Return ONLY the SQL query without any markdown formatting or explanations."},
                {"role": "user", "content": prompt}
            ]
        )
        
        sql = response.choices[0].message.content.strip()
        clean_sql = self.clean_sql(sql)
        logger.info(f"Generated SQL (Attempt {attempt}): {clean_sql}")
        return clean_sql

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    logger.info(f"Executing SQL: {sql}")
                    cur.execute(sql)
                    columns = [desc[0] for desc in cur.description]
                    results = cur.fetchall()
                    result_count = len(results)
                    logger.info(f"Query returned {result_count} results")
                    return [dict(zip(columns, row)) for row in results]
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise Exception(f"Database error: {str(e)}")

    def generate_insights(self, results: List[Dict[str, Any]], query: str) -> str:
        insight_prompt = f"""
        Analyze these query results for the question: "{query}"
        
        Data:
        {json.dumps(results, indent=2, default=str)}
        
        Provide:
        1. Key metrics and their significance
        2. Notable patterns or trends
        3. Business implications
        4. Potential action items
        
        Keep insights concise and actionable.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a business analyst providing concise, actionable insights."},
                {"role": "user", "content": insight_prompt}
            ]
        )
        
        insights = response.choices[0].message.content
        logger.info(f"Generated insights: {insights}")
        return insights

    def attempt_query_execution(self, natural_query: str, attempt: int = 1) -> Optional[Dict[str, Any]]:
        try:
            sql = self.enhance_query(natural_query, attempt)
            results = self.execute_query(sql)
            
            if not results:
                logger.warning(f"Attempt {attempt}: Query returned no results")
                raise Exception("Query returned no results")
            
            insights = self.generate_insights(results, natural_query)
            
            return {
                "query": natural_query,
                "sql": sql,
                "results": results,
                "insights": insights,
                "attempt": attempt,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            if attempt < self.MAX_RETRIES:
                logger.warning(f"Attempt {attempt} failed with error: {str(e)}. Retrying...")
                time.sleep(self.RETRY_DELAY)
                return None
            else:
                raise e

    def process_query(self, natural_query: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing query: {natural_query}")
            
            if not all([self.db_config["host"], self.db_config["database"], 
                       self.db_config["user"], self.db_config["password"]]):
                raise Exception("Database configuration is incomplete")
                
            if not os.getenv("OPENAI_API_KEY"):
                raise Exception("OpenAI API key is not set")
            
            last_error = None
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    result = self.attempt_query_execution(natural_query, attempt)
                    if result:
                        logger.info(f"Query processed successfully on attempt {attempt}")
                        return result
                except Exception as e:
                    last_error = e
                    logger.error(f"Attempt {attempt} failed: {str(e)}")
                    
            error_msg = f"All {self.MAX_RETRIES} attempts failed. Last error: {str(last_error)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "query": natural_query,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing query: {error_msg}")
            return {
                "error": error_msg,
                "query": natural_query,
                "timestamp": datetime.now().isoformat()
            }
