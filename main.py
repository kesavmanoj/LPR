import sys
from src.train_corner import train as train_corner
from src.test_corner import test_and_save_csv as test_corner


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train_corner|test_corner]")
        sys.exit(1)
    mode = sys.argv[1].lower()
    if mode == "train_corner":
        train_corner()
    elif mode == "test_corner":
        test_corner()
    else:
        print("Invalid mode. Please choose 'train_corner' or 'test_corner'.")

if __name__ == "__main__":
    main()
