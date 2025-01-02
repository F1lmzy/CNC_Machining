import os
import shutil
from pathlib import Path

def reorganize_data(source_base_path, target_base_path):
    """
    Reorganize data from OP00-OP10 into OPTrain directory
    
    Args:
        source_base_path: Base path containing OPXX directories
        target_base_path: Target path for OPTrain directory
    """
    # Create OPTrain directory structure
    target_good_path = os.path.join(target_base_path, 'good')
    target_bad_path = os.path.join(target_base_path, 'bad')
    
    os.makedirs(target_good_path, exist_ok=True)
    os.makedirs(target_bad_path, exist_ok=True)
    
    # Process OP00 to OP10
    for op_num in range(11):  # 0 to 10
        op_dir = f'OP{op_num:02d}'
        source_op_path = os.path.join(source_base_path, op_dir)
        
        if not os.path.exists(source_op_path):
            print(f"Skipping {op_dir} - directory not found")
            continue
            
        # Process good files
        source_good_path = os.path.join(source_op_path, 'good')
        if os.path.exists(source_good_path):
            for file in os.listdir(source_good_path):
                if file.endswith('.h5'):
                    source_file = os.path.join(source_good_path, file)
                    target_file = os.path.join(target_good_path, f"{op_dir}_{file}")
                    shutil.copy2(source_file, target_file)
                    print(f"Copied {file} to {target_file}")
        
        # Process bad files
        source_bad_path = os.path.join(source_op_path, 'bad')
        if os.path.exists(source_bad_path):
            for file in os.listdir(source_bad_path):
                if file.endswith('.h5'):
                    source_file = os.path.join(source_bad_path, file)
                    target_file = os.path.join(target_bad_path, f"{op_dir}_{file}")
                    shutil.copy2(source_file, target_file)
                    print(f"Copied {file} to {target_file}")

if __name__ == "__main__":
    # Define paths
    source_base = "./data/M01"
    target_base = "./data/M01/OPTrain"
    
    # Execute reorganization
    print("Starting data reorganization...")
    reorganize_data(source_base, target_base)
    print("Data reorganization complete!")
    
    # Print summary
    good_files = len(os.listdir(os.path.join(target_base, 'good')))
    bad_files = len(os.listdir(os.path.join(target_base, 'bad')))
    print(f"\nSummary:")
    print(f"Total good files: {good_files}")
    print(f"Total bad files: {bad_files}") 