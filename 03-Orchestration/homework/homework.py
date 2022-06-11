import datetime
import pickle

import pandas as pd
from prefect import flow, task, get_run_logger
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


@task()
def get_paths(date):
    logger = get_run_logger()
    if date is None:
        date = datetime.datetime.now()
    # get month from datetime object
    month = date.month
    train_path = f"./data/fhv_tripdata_2021-{month - 2:02d}.parquet"
    val_path = f"./data/fhv_tripdata_2021-{month - 1:02d}.parquet"

    logger.info(f"The path of training data is {train_path}")
    logger.info(f"The path of validation data is {val_path}")

    return train_path, val_path


def pkl_dump(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


@flow(name='homework')
def main(date=None):
    train_path, val_path = get_paths(date).result()

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path).result()
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path).result()
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)

    # Save the model
    date = date.strftime("%Y-%m-%d")
    model_name = f"artifacts/model-{date}.bin"
    dict_vect_name = f"artifacts/dv-{date}.bin"

    pkl_dump(lr, model_name)
    pkl_dump(dv, dict_vect_name)


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    name="scheduled-deployment",
    flow=main,
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="Asia/Colombo"
    ),
    tags=["Cron Scheduled"],
    flow_runner=SubprocessFlowRunner()
)

if __name__ == '__main__':
    date_time_str = "2021-08-15"
    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
    main(date_time_obj)
