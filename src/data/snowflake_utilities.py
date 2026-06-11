# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:08:21 2026

@author: flent
"""

import os
import re
import datetime as dt
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from snowflake.connector.pandas_tools import logger
import logging

logger.setLevel(logging.DEBUG)

SNOWFLAKE_ACCOUNT = "LW64461-NRDATA"
SNOWFLAKE_USER = "SVC_USERTEAMWILHELM"

# Optional: set this if your RA user needs a specific role.
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")

# Your RSA private key file (.p8)
SNOWFLAKE_PRIVATE_KEY_FILE = os.getenv(
    "SNOWFLAKE_PRIVATE_KEY_FILE", r'',
)

# Only set this environment variable if the .p8 file is encrypted.
SNOWFLAKE_PRIVATE_KEY_FILE_PWD = os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE_PWD")


def snowflake_conn(
    warehouse,
    database,
    schema,
    autocommit=True,
    account=SNOWFLAKE_ACCOUNT,
    user=SNOWFLAKE_USER,
    role=SNOWFLAKE_ROLE,
    private_key_file=SNOWFLAKE_PRIVATE_KEY_FILE,
    private_key_file_pwd=SNOWFLAKE_PRIVATE_KEY_FILE_PWD,
):
    """Snowflake Python Connector connection using RA key-pair authentication."""
    if not private_key_file:
        raise ValueError("private_key_file must be set for RA key authentication.")

    connect_args = {
        "account": account,
        "user": user,
        "database": database,
        "schema": schema,
        "warehouse": warehouse,
        "autocommit": autocommit,
        "authenticator": "SNOWFLAKE_JWT",
        "private_key_file": os.path.expandvars(os.path.expanduser(private_key_file)),
    }

    if role:
        connect_args["role"] = role

    # Only include passphrase if the private key is encrypted
    if private_key_file_pwd:
        connect_args["private_key_file_pwd"] = private_key_file_pwd

    ctx = snowflake.connector.connect(**connect_args)
    cs = ctx.cursor()
    return cs, ctx


def read_sql(
    query: str,
    return_df=True,
    to_lower: bool = False,
    warehouse='WHSE_TEAM_WILHELM_001',
    database='DB_TEAM_WILHELM_001',
    schema='PUBLIC',
    num_statements=None,
):
    """
    Send SQL query to database via direct snowflake adapter.
    Default return results as a Pandas dataframe.
    param: query: the query to execute
    param: return_df: whether to return a pandas dataframe
    param: to_lower: whether to lower case column names in dataframe
    """
    cs, ctx = snowflake_conn(warehouse=warehouse, database=database, schema=schema)
    try:
        cs.execute(query, num_statements=num_statements)
        if return_df:
            result = cs.fetch_pandas_all()
            if to_lower:
                result = result.rename(columns=str.lower)
        else:
            result = cs.fetchall()
        return result
    finally:
        cs.close()
        ctx.close()


def dates_to_sql(dates):
    """Convert a list of dates to a SQL-compatible string for IN clauses."""
    if isinstance(dates, (pd.Timestamp, dt.date, dt.datetime)):
        dates = [dates]
    date_strs = [pd.Timestamp(d).strftime("'%Y-%m-%d'") for d in dates]
    return "(" + ", ".join(date_strs) + ")"
