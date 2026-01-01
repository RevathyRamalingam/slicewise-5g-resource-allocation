import pytest

def test_load_dataset(tmp_path, monkeypatch):
    #create a mock csv file
    df = pd.DataFrame({"Unnamed: 0":[1,2],
    "Packet loss %": [1,2,3],
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