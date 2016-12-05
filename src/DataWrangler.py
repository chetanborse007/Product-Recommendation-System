#!/usr/bin/python
# -*- coding: utf-8 -*- 
'''
@File:           DataWrangler.py
@Description:    PySpark application for cleaning and preprocessing customer
                 and product purchase data.
@Author:         Chetan Borse
@EMail:          cborse@uncc.edu
@Created on:     12/04/2016
@Usage:          python src/InvertedIndexer.py
                    -i "input/train.csv"
                    -o "input/preprocessed_train.csv"
@python_version: 2.7
===============================================================================
'''


import os
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Global variables
CustomerAttribute = {}


def SetGlobalUniqueValue(uniqueValues, columnName):
    """
    Set global unique values.

    @param: uniqueValues  => Categorical values.
    @param: columnName    => Column name.

    @returns:             => None.
    """
    global CustomerAttribute
    CustomerAttribute[columnName] = dict(zip(uniqueValues, 
                                             range(1, len(uniqueValues)+1)))


def DataWrangler(**args):
    """
    PySpark application for cleaning and preprocessing customer
    and product purchase data.
    """
    # Read arguments
    input  = args['input']
    output = args['output']

    # Check whether input csv exists
    if not input or not os.path.isfile(input):
        print "Input csv does not exist!"
        return -1

    # Read input csv into panda's data frame
    df = pd.read_csv(input,
                     na_values=["-999999", "NA", "         NA"])

    # Transform date and time format
    df["fecha_dato"]     = pd.to_datetime(df["fecha_dato"], format="%Y-%m-%d")
    df["fecha_alta"]     = pd.to_datetime(df["fecha_alta"], format="%Y-%m-%d")
    df["ult_fec_cli_1t"] = pd.to_datetime(df["ult_fec_cli_1t"], format="%Y-%m-%d")
    
    # If 'ult_fec_cli_1t' i.e. Last date as primary customer, is empty;
    # then insert default date and time.
    df.loc[df.ult_fec_cli_1t.isnull(), "ult_fec_cli_1t"] = \
                            pd.to_datetime("2015-01-28", format="%Y-%m-%d")

    # Normalize age distribution.
    # Move the outliers to the mean of the closest distribution.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[df.age<18, "age"]  = df.loc[(df.age>=18) & (df.age<=30), "age"] \
                                  .mean(skipna=True)
    df.loc[df.age>100, "age"] = df.loc[(df.age>=30) & (df.age<=100), "age"] \
                                  .mean(skipna=True)
    df["age"].fillna(df["age"].mean(), inplace=True)
    df["age"] = df["age"].astype(int)

    # Fill missing values in new customer index.
    df.loc[df["ind_nuevo"].isnull(), "ind_nuevo"] = 1

    # Fill missing values in customer seniority with minimum seniority
    df.antiguedad = pd.to_numeric(df.antiguedad, errors="coerce")
    df.loc[df.antiguedad.isnull(), "antiguedad"] = df.antiguedad.min()
    df.loc[df.antiguedad<0, "antiguedad"]        = 0

    # Fill missing values in joining dates with median value
    joining_dates       = df.loc[:, "fecha_alta"].sort_values().reset_index()
    median_joining_date = int(np.median(joining_dates.index.values))
    df.loc[df.fecha_alta.isnull(), "fecha_alta"] = \
                            joining_dates.loc[median_joining_date, "fecha_alta"]

    # Fill missing values in 'indrel'
    df.loc[df.indrel.isnull(), "indrel"] = 1

    # Drop address type and province code columns.
    # These columns are not useful because the name of the province exists in 'nomprov'.
    df.drop(["tipodom", "cod_prov"], axis=1, inplace=True)

    # Fill missing values in activity index with median value
    df.loc[df.ind_actividad_cliente.isnull(), "ind_actividad_cliente"] = \
                            df["ind_actividad_cliente"].median()

    # Fix the unicode character 'ñ' in 'A Coruña'.
    df.loc[df.nomprov=="CORU\xc3\x91A, A", "nomprov"] = "CORUNA, A"
    df.loc[df.nomprov.isnull(), "nomprov"]            = "UNKNOWN"

    # Fill the missing values in gross income of the household.
    # Fill missing values with median value of gross incomes for a
    # particular region.
    grouped     = df.groupby("nomprov") \
                    .agg({"renta": lambda x: x.median(skipna=True)}) \
                    .reset_index()
    new_incomes = pd.merge(df, grouped, how="inner", on="nomprov") \
                    .loc[:, ["nomprov", "renta_y"]]
    new_incomes = new_incomes.rename(columns={"renta_y": "renta"}) \
                             .sort_values("renta") \
                             .sort_values("nomprov")
    df.sort_values("nomprov", inplace=True)
    df          = df.reset_index()
    new_incomes = new_incomes.reset_index()
    df.loc[df.renta.isnull(), "renta"] = new_incomes.loc[df.renta.isnull(), "renta"] \
                                                    .reset_index()
    df.loc[df.renta.isnull(), "renta"] = df.loc[df.renta.notnull(), "renta"] \
                                           .median()
    df.sort_values(by="fecha_dato", inplace=True)

    # Fill missing values in payroll and pension.
    df.loc[df.ind_nomina_ult1.isnull(), "ind_nomina_ult1"]     = 0
    df.loc[df.ind_nom_pens_ult1.isnull(), "ind_nom_pens_ult1"] = 0

    # Fill missing values in deceased index.
    df.loc[df.indfall.isnull(), "indfall"] = "N"

    # Fill missing values in customer relation type with most common value
    df.loc[df.tiprel_1mes.isnull(), "tiprel_1mes"] = "A"
    df.tiprel_1mes = df.tiprel_1mes.astype("category")

    # Transform customer relation type to numerical values
    customerType = {1.0: "1",
                    "1.0": "1",
                    "1": "1",
                    "3.0": "3",
                    "P": "P",
                    3.0: "3",
                    2.0: "2",
                    "3": "3",
                    "2.0": "2",
                    "4.0": "4",
                    "4": "4",
                    "2": "2"}
    df.indrel_1mes.fillna("P", inplace=True)
    df.indrel_1mes = df.indrel_1mes.apply(lambda x: customerType.get(x, x))
    df.indrel_1mes = df.indrel_1mes.astype("category")

    # Set 'UNKNOWN' for missing values in every other column
    columns        = df.select_dtypes(include=["object"])
    missingColumns = [c for c in columns if columns[c].isnull().any()]
    for m in missingColumns:
        df.loc[df[m].isnull(), m] = "UNKNOWN"

    # Convert product columns into integer form.
    products = df.iloc[:1, ].filter(regex="ind_+.*ult.*").columns.values
    for p in products:
        df[p] = df[p].astype(int)

    # Transform all date and time columns into time difference 
    # with respect to some base date and time.
    baseDate = pd.to_datetime("1902-01-01", format="%Y-%m-%d")
    df['fecha_dato'] = (df['fecha_dato'] - baseDate) / \
                       np.timedelta64(1, 'D')
    df['fecha_alta'] = (df['fecha_alta'] - baseDate) / \
                       np.timedelta64(1, 'D')
    df['ult_fec_cli_1t'] = (df['ult_fec_cli_1t'] - baseDate) / \
                       np.timedelta64(1, 'D')

    # Drop index column.
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # Transform attribute columns into numerical form.
    columns = list(df.columns.values)
    for c in columns:
        if c in ["ind_empleado", \
                 "pais_residencia", \
                 "sexo", \
                 "indrel_1mes", \
                 "tiprel_1mes", \
                 "indresi", \
                 "indext", \
                 "conyuemp", \
                 "canal_entrada", \
                 "indfall", \
                 "nomprov", \
                 "segmento"]:
            uniqueValues = df[c].unique();print c;print uniqueValues;print '\n'
            SetGlobalUniqueValue(uniqueValues, c)

            df[c] = df[c].apply(lambda x: CustomerAttribute.get(c).get(x))

    print CustomerAttribute
    # Drop few columns, which are not useful.
    df.drop(["fecha_dato", "fecha_alta", "ult_fec_cli_1t", "renta"], axis=1, inplace=True)
    
    # Finally, save data frame into csv.
    df.to_csv(output, index=False)


if __name__ == "__main__":
    """
    Entry point.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description='Data Wrangler Application',
                                     prog='python src/DataWrangler.py \
                                           -i <input> \
                                           -o <output>')

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input csv for cleaning and preprocessing.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Preprocessed csv.")

    # Read user inputs
    args = vars(parser.parse_args())

    # Run Data Wrangler Application
    DataWrangler(**args)

