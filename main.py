import sys
from src.train import train
from src.test import test

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|test]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    if mode == "train":
        train()
    elif mode == "test":
        test()
    else:
        print("Invalid mode. Please choose 'train' or 'test'.")

if __name__ == "__main__":
    main()
