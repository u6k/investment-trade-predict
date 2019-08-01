import os
import io
import joblib
import pandas as pd
import boto3
from keras.models import model_from_json
from PIL import Image
import numpy as np
import tarfile
import tempfile
import gzip as gz


def get_client():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_S3_SECRET_KEY"],
        endpoint_url=os.environ["AWS_S3_ENDPOINT_URL"]
    )

    return s3


def read_dataframe(s3_bucket, s3_key, gzip=True, **kwargs):
    s3 = get_client()
    if gzip:
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key+".gz")
        with io.BytesIO(obj["Body"].read()) as buf:
            data = gz.decompress(buf.getvalue())
            with io.BytesIO(data) as buf_data:
                df = pd.read_csv(buf_data, **kwargs)
    else:
        obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        df = pd.read_csv(obj["Body"], **kwargs)

    return df


def write_dataframe(df, s3_bucket, s3_key, gzip=True):
    with io.StringIO() as buf:
        df.to_csv(buf)
        s3 = get_client()
        if gzip:
            s3.put_object(
                Bucket=s3_bucket,
                Key=s3_key+".gz",
                Body=io.BytesIO(gz.compress(buf.getvalue().encode()))
            )
        else:
            s3.put_object(
                Bucket=s3_bucket,
                Key=s3_key,
                Body=io.BytesIO(buf.getvalue().encode())
            )


def write_sklearn_model(clf, s3_bucket, s3_key):
    with io.BytesIO() as buf:
        joblib.dump(clf, buf, compress=9)
        s3 = get_client()
        s3.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=io.BytesIO(buf.getvalue())
        )


def read_sklearn_model(s3_bucket, s3_key):
    s3 = get_client()
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    with io.BytesIO(obj["Body"].read()) as buf:
        clf = joblib.load(buf)

    return clf


def write_keras_model(model, s3_bucket, s3_key_prefix):
    s3 = get_client()
    s3.put_object(
        Bucket=s3_bucket,
        Key=s3_key_prefix + ".json",
        Body=model.to_json()
    )

    with io.BytesIO() as buf:
        model.save_weights(buf)

        s3.put_object(
            Bucket=s3_bucket,
            Key=s3_key_prefix+".hdf5",
            Body=io.BytesIO(buf.getvalue())
        )


def read_keras_model(s3_bucket, s3_key_prefix):
    s3 = get_client()
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key_prefix+".json")
    model = model_from_json(obj["Body"].read())

    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key_prefix+".hdf5")
    with io.BytesIO(obj["Body"].read()) as buf:
        model.load_weights(buf)

    return model


def write_images(np_array, s3_bucket, s3_key, file_name_prefix):
    with io.BytesIO() as tar_buf:
        tar = tarfile.open(fileobj=tar_buf, mode="w:gz")
        for i, np_2d_array in enumerate(np_array):
            with io.BytesIO() as buf:
                img = Image.fromarray(np.uint8(np_2d_array))
                img.save(buf, format="PNG")

                tar_info = tarfile.TarInfo(name=f"{file_name_prefix}.{i}.png")
                tar_info.size = buf.getbuffer().nbytes

                tar.addfile(tar_info, fileobj=io.BytesIO(buf.getvalue()))

        tar.close()

        s3 = get_client()
        s3.put_object(
            Bucket=s3_bucket,
            Key=s3_key,
            Body=io.BytesIO(tar_buf.getvalue())
        )


def extract_images(df_target_train, df_target_test, s3_bucket, s3_key, file_name_prefix):
    tmp_dir = tempfile.TemporaryDirectory()

    s3 = get_client()
    obj = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    with io.BytesIO(obj["Body"].read()) as buf:
        tar = tarfile.open(fileobj=buf)

        for index in df_target_train.index:
            os.makedirs(f"{tmp_dir.name}/data/train/{df_target_train.at[index, 'predict_target']}/", exist_ok=True)

            buf_png = tar.extractfile(f"{file_name_prefix}.{index}.png")
            open(f"{tmp_dir.name}/data/train/{df_target_train.at[index, 'predict_target']}/{file_name_prefix}.{index}.png", "wb").write(buf_png.read())

    return tmp_dir
