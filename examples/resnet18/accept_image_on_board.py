"""CD acceptance of the default real photograph through the compiled model."""
import sys
from run_on_board import main

if __name__ == '__main__':
    raise SystemExit(main(['--expected-class','Samoyed',*sys.argv[1:]]))
