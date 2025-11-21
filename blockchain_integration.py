import hashlib
import os
from pathlib import Path
from datetime import datetime

def generate_file_hash(file_path):
    """Generate SHA-256 hash of a file"""
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def generate_project_hash(project_dir='.'):
    """Generate hash of entire project"""
    sha256_hash = hashlib.sha256()
    
    # Files to exclude
    exclude_dirs = {'.git', '.venv', '__pycache__', '.pytest_cache', 'node_modules'}
    exclude_extensions = {'.pyc', '.pyo', '.egg-info'}
    
    try:
        for root, dirs, files in os.walk(project_dir):
            # Remove excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # Sort for consistent hashing
            files.sort()
            
            for file in files:
                # Skip excluded extensions and large files
                if any(file.endswith(ext) for ext in exclude_extensions):
                    continue
                if file.endswith('.csv') and os.path.getsize(os.path.join(root, file)) > 100 * 1024 * 1024:
                    continue
                
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        sha256_hash.update(f.read())
                except Exception as e:
                    print(f"Warning: Could not hash {file_path}: {e}")
        
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error generating project hash: {e}")
        return None

def create_blockchain_record(project_hash, description="Sentiment Analysis Project"):
    """Create a blockchain-ready record"""
    record = {
        "timestamp": datetime.now().isoformat(),
        "project_hash": project_hash,
        "description": description,
        "version": "1.0.0",
        "network": "ethereum",
        "status": "pending_smart_contract"
    }
    return record

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("BLOCKCHAIN INTEGRATION - PROJECT HASH GENERATION")
    print("=" * 60)
    
    # Generate project hash
    print("\nGenerating project hash...")
    project_hash = generate_project_hash()
    
    if project_hash:
        print(f"\n✓ Project Hash (SHA-256): {project_hash}")
        print(f"  Hash Length: {len(project_hash)} characters")
        
        # Create blockchain record
        record = create_blockchain_record(project_hash)
        
        print(f"\n✓ Blockchain Record Created:")
        for key, value in record.items():
            print(f"  {key}: {value}")
        
        # Save hash to file
        hash_file = "project_hash.txt"
        with open(hash_file, 'w') as f:
            f.write(f"Project Hash: {project_hash}\n")
            f.write(f"Generated: {record['timestamp']}\n")
            f.write(f"Description: {record['description']}\n")
        
        print(f"\n✓ Hash saved to '{hash_file}'")
        print("\n" + "=" * 60)
        print("Ready for smart contract integration!")
        print("=" * 60)
    else:
        print("Failed to generate project hash")
