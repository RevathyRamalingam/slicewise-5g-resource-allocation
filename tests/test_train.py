import pytest
import pandas as pd
import src.train as train
from datetime import datetime
from zoneinfo import ZoneInfo

def test_load_dataset1(tmp_path, monkeypatch):
    #create a mock csv file
    df = pd.DataFrame({"Unnamed: 0":[1,2],
    "Packet loss %": [1,2],
    "Avg Delay (ms)":[1,2]
    })

    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    fake_file = tmp_path / "train.py"
    fake_file.touch()

    monkeypatch.setattr(train,"__file__", str(fake_file))
    result = train.load_dataset(csv_path)

    assert "packet_loss_pct" in result.columns
    assert "avg_delay_ms" in result.columns

def test_load_dataset2(tmp_path, monkeypatch):
    #create a mock csv file
    df = pd.DataFrame({"valid_column_name":[1,2],
    "SOME_COLUMN":[1,2]
    })

    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    fake_file = tmp_path / "train.py"
    fake_file.touch()

    monkeypatch.setattr(train,"__file__", str(fake_file))
    result = train.load_dataset(csv_path)

    assert "valid_column_name" in result.columns
    assert "some_column" in result.columns

def test_convert_timestamp_to_networkload_tc1():
    """Test off-peak load at 13:00 Rome time."""
    rome_dt = datetime(2026, 1, 2, 13, 0, 0, tzinfo=ZoneInfo("Europe/Rome"))
    off_peak_timestamp = int(rome_dt.timestamp() * 1000)

    hour_op, load_op = train.convert_timestamp_to_networkLoad(off_peak_timestamp)

    assert load_op == "off-peak"
    assert hour_op == 13

def test_convert_timestamp_to_networkload_tc2():
    """Test off-peak load at 13:00 Rome time."""
    rome_dt = datetime(2026, 1, 2, 2, 0, 0, tzinfo=ZoneInfo("Europe/Rome"))
    off_peak_timestamp = int(rome_dt.timestamp() * 1000)

    hour_op, load_op = train.convert_timestamp_to_networkLoad(off_peak_timestamp)

    assert load_op == "night"
    assert hour_op == 2

def test_convert_timestamp_to_networkload_tc3():
    """Test off-peak load at 13:00 Rome time."""
    rome_dt = datetime(2026, 1, 2, 21, 0, 0, tzinfo=ZoneInfo("Europe/Rome"))
    off_peak_timestamp = int(rome_dt.timestamp() * 1000)

    hour_op, load_op = train.convert_timestamp_to_networkLoad(off_peak_timestamp)

    assert load_op == "peak"
    assert hour_op == 21