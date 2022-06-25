#!/usr/bin/env python
# coding: utf-8

import sys

import os
import pickle

import numpy as np
import pandas as pd

categorical = ['PUlocationID', 'DOlocationID']

def load_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr



def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


def run():
    year = int(sys.argv[1])  # 2021
    month = int(sys.argv[2])  # 2
    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f"predictions_{year:04d}-{month:02d}_fvh_data.parquet"

    print("Reading data from ", input_file)
    df = read_data(input_file)

    print("Load Model")
    dv, lr = load_model()

    print("Preparing features")
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    print("Predicting")
    y_pred = lr.predict(X_val)
    mean = round(np.mean(y_pred), 2)
    print(f"The mean duration of prediction is {mean}")

    print("Writing to ", output_file)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    data = {
        'ride_id': df['ride_id'].values,
        'predictions': y_pred
    }
    df_result = pd.DataFrame(data)
    if not os.path.exists('output'):
        os.makedirs('output')
    df_result.to_parquet(
        os.path.join("output", output_file),
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == '__main__':
    run()
