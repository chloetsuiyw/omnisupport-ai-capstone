from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_primary_dataset_is_parquet_and_has_ten_shards():
    shards = sorted((ROOT / 'data' / 'raw').glob('support_records_part_*.parquet'))
    assert len(shards) == 10
    assert not list((ROOT / 'data' / 'raw').glob('support_records_part_*.csv.gz'))


def test_required_supporting_assets_exist():
    assert (ROOT / 'data' / 'subsets' / 'transformer_finetune_10000.parquet').exists()
    assert (ROOT / 'data' / 'agent_store' / 'orders.csv').exists()
    assert (ROOT / '.github' / 'workflows' / 'tests.yml').exists()
    assert (ROOT / 'monitoring' / 'monitoring_summary_template.csv').exists()
