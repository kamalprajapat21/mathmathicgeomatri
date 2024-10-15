import hashlib
import time

# Define the block structure
class Block:
    def __init__(self, index, previous_hash, timestamp, data, proof):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.proof = proof
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.proof}"
        return hashlib.sha256(block_string.encode()).hexdigest()

# Define the blockchain structure
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, "0", time.time(), "Genesis Block", 0)
        self.chain.append(genesis_block)

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data, proof):
        latest_block = self.get_latest_block()
        new_block = Block(len(self.chain), latest_block.hash, time.time(), data, proof)
        self.chain.append(new_block)

    def proof_of_work(self, previous_proof):
        new_proof = 1
        while True:
            hash_value = hashlib.sha256(str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_value[:4] == "0000":  # Mining difficulty
                return new_proof
            new_proof += 1

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            if current_block.hash != current_block.calculate_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

# Usage
blockchain = Blockchain()

# Simulate adding blocks with proof of work
previous_proof = 0
for i in range(1, 4):
    proof = blockchain.proof_of_work(previous_proof)
    blockchain.add_block(f"Transaction data {i}", proof)
    previous_proof = proof

# Output the blockchain
for block in blockchain.chain:
    print(f"Block {block.index} - Hash: {block.hash}, Previous Hash: {block.previous_hash}, Data: {block.data}, Proof: {block.proof}")
