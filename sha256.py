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
    14.1347251417346937904572519835624702707842571156992431756855674601499634298092567649490103931715610127,
    21.0220396387715549926284795938969027773343405249027817546295204035875985860688907997102519702967588,
    25.0108575801456887632137909925628218186595496725579966724965420066164767192309727364028112763273236,
    30.4248761258595132103118975305840913201815600237154401809621460369933293797578227279007793582571663,
    32.9350615877391896906623600823440582899375977712868692923413173477117453704921616388168681830017915,
    37.5861781588256712572177634807053328214055973508307932183330011136221490896185372477865753394563538,
    40.9187190121474951873981269146332543957261659627772795361613036672532805287100767293430915712340554,
    43.3270732809141962702914104591200792340214963137457070347171703241310825513907645297001669592511709,
    48.0051508811671597322521711509373350542180354878661137904574536117218678739694837582318122179100390,
    49.7738324776723021819167846785639900664457872323276309416759171941018442182086714585163058645754089,
    52.9703214777143850208988044289177302443015010297414087705339191303996352019789349883450227660746723,
    56.4462476972982182787492405583122814911016303461649657535261177992187916691009390610426208055893014,
    59.3470440026023530796536486749921313487092253933675868500656004477428142795833542590876143938180964,
    60.8317785257335330251793247778025376688949947864404452606910098066560600715321824633321613219961370,
    65.1125440481846765772148374180743750244141347475762456361472293583147411629975900695481026873962560,
    67.0798105294941733616834377739900688281520967458358778948293303657980647753181892416459322452298550,
    69.5464017111731015111476882177820133407723270915781664956018766179797591180130122658743570437307032,
    72.0671576744819067812255666840185545507228873428026860832061800645132217765372225525961747018711465,
    75.7046906990833929726325755133324106793124693557573157443037444398938243510879262551491319021478882,
    77.1448400688747861576989357999082688891579651646149416475234584509570100678897511610848339826013480
]
        
    def zeta_critical_line(self, t: float, terms: int = 1000) -> complex:
        """
        Direct computation of zeta(1/2 + it) using Riemann-Siegel formula approximation
        """
        s = 0.01 + 1j * t
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
            data = f"GeorgeW{nonce}{str(abs(zeta_val.real)).replace('.', '')[:64]}".encode()
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
                
                for i in range(5000000):
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
    for i in range(17):
        print("Starting Riemann zeta function based SHA-256 mining")
        print(f"Using {multiprocessing.cpu_count()} CPU cores")
        miner = ZetaMiner()
        miner.mine(difficulty=i+1)
        
