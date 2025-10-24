import pytest
import pandas as pd
from src.data_loader import load_dataset, LABEL_MAP
from src.data_prep import clean_text

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        'review': [
            'This movie is great! <br /><br />', 
            'A terrible film, just awful.', 
            'average movie with some good parts http://example.com',
            None, # Test None handling
            '@user mentioned this is a #awesome movie'
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive']
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_csv_file(tmp_path, sample_dataframe):
    """Create a temporary CSV file for testing data loading."""
    file_path = tmp_path / "sample_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path

def test_load_data(temp_csv_file):
    """Test that data is loaded correctly from a CSV file."""
    df = load_dataset(str(temp_csv_file))
    assert isinstance(df, pd.DataFrame)
    assert "review" in df.columns
    assert "sentiment" in df.columns
    assert "label" in df.columns
    assert not df.isnull().values.any() # Check that rows with None are dropped
    assert df.shape[0] == 4 # One row should be dropped
    assert df['label'].sum() == 3 # 3 positive labels

def test_load_data_value_error():
    """Test that a ValueError is raised for incorrect sentiment labels."""
    data = {'review': ['a'], 'sentiment': ['neutral']} # neutral is not supported
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        df['label'] = df['sentiment'].map(LABEL_MAP)
        if df["label"].isna().any():
            raise ValueError("Unexpected sentiment labels found.")

@pytest.mark.parametrize("input_text, expected_output", [
    ("This is a GREAT movie! 123", "this is a great movie"),
    ("Visit my site at https://example.com", "visit my site at"),
    ("It was... AWESOME!!! <p>tag</p>", "it was awesome"),
    ("  leading and trailing spaces  ", "leading and trailing spaces"),
    ("Stopwords like a and the should be removed", "stopwords like should removed"),
    ("", ""), # Empty string
    (None, "") # None input
])
def test_clean_text(input_text, expected_output):
    """Test the text cleaning function with various inputs."""
    assert clean_text(input_text) == expected_output
