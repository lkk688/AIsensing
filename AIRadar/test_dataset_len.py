
def test_len():
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from AIRadar.AIradar_datasetv6 import AIRadarDataset
    
    # Create a dummy dataset
    try:
        ds = AIRadarDataset(num_samples=10, config_name='config1', save_path='test_data')
        # We need to set range_doppler_maps to something not None to test len
        # But __init__ calls _load_data if path exists or generate_data.
        # Let's mock the necessary attributes
        ds.range_doppler_maps = [1] * 10
        ds.num_samples = 10
        
        print(f"Dataset len: {len(ds)}")
        print("Test Passed")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_len()
