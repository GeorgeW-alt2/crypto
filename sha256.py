import hashlib
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Optional
import numpy as np
from scipy.special import gamma

class ZetaMiner:
    def __init__(self):
        # Known first few zeros of zeta function
        self.zeros = [
            14,
            21,
            25,
            30,
            33,
            38,
            41,
            43,
            48,
            50,
            53,
            56,
            59,
            61,
            65,
            67,
            70,
            72,
            76,
            77
        ]
        
    def zeta_critical_line(self, t: float, terms: int = 1000) -> complex:
        """
        Direct computation of zeta(1/2 + it) using Riemann-Siegel formula approximation
        """
        s = 0.5 + 1j * t
        N = int(np.sqrt(t/(2*np.pi)))
        
        # Main sum
        result = 0
        for n in range(1, N + 1):
            result += np.exp(np.log(n) * (-s))
            
        # Phase factor
        theta = -np.log(np.pi)/2 * s + np.log(gamma(s/2))
        Z = np.exp(1j * theta) * result
        
        return Z

    def mine_chunk(self, start_nonce: int, chunk_size: int, difficulty: int, t: float) -> Optional[Tuple[int, str, bytes]]:
        target = '0' * difficulty
        zeta_val = self.zeta_critical_line(t)
        
        for nonce in range(start_nonce, start_nonce + chunk_size):
            # Combine nonce with zeta value for mining
            data = f"GeorgeW{nonce}{zeta_val.real:.6f}{zeta_val.imag:.6f}".encode()
            hash_result = hashlib.sha256(data).hexdigest()
            
            if hash_result.startswith(target):
                return nonce, hash_result, data
                
        return None

    def mine(self, difficulty: int = 4, threads: Optional[int] = None):
        threads = threads or multiprocessing.cpu_count()
        chunk_size = 1000000
        start_time = time.time()
        total_hashes = 0
        last_print = start_time
        print_interval = 1.0
        
        # Use known zeros of zeta function
        for t in self.zeros:
            print(f"\nMining with zeta zero t = {t}")
            zeta_val = self.zeta_critical_line(t)
            print(f"ζ(1/2 + {t}i) ≈ {zeta_val}")
            
            with ProcessPoolExecutor(max_workers=threads) as executor:
                nonce = 0
                futures = []
                
                while True:
                    # Keep threads busy
                    while len(futures) < threads:
                        future = executor.submit(self.mine_chunk, nonce, chunk_size, difficulty, t)
                        futures.append((future, nonce))
                        nonce += chunk_size
                    
                    # Check results
                    for future, start_nonce in futures[:]:
                        if future.done():
                            futures.remove((future, start_nonce))
                            total_hashes += chunk_size
                            
                            now = time.time()
                            if now - last_print >= print_interval:
                                elapsed = now - start_time
                                rate = total_hashes / elapsed
                                print(f"Hash rate: {rate:,.0f} H/s ({total_hashes:,} hashes)", flush=True)
                                last_print = now
                            
                            result = future.result()
                            if result:
                                found_nonce, hash_result, data = result  # Properly unpack three values
                                elapsed = time.time() - start_time
                                final_rate = total_hashes / elapsed
                                print(f"\nFound solution after {elapsed:.2f} seconds")
                                print(f"Using zeta zero t = {t}")
                                print(f"ζ(1/2 + {t}i) ≈ {zeta_val}")
                                print(f"Final hash rate: {final_rate:,.0f} H/s")
                                print(f"Nonce: {found_nonce}")
                                print(f"Hash: {hash_result}")
                                print(f"Data: {data.decode()}")  # Decode bytes to string for display
                                return

if __name__ == "__main__":
    print("Starting Riemann zeta function based SHA-256 mining")
    print(f"Using {multiprocessing.cpu_count()} CPU cores")
    miner = ZetaMiner()
    miner.mine(difficulty=5)
