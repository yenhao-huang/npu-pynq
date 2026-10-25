"""Execute a board command; read sudo password only from workspace .env.

SSH uses the existing key. The password travels only on stdin to sudo -S;
never in arguments, files, stdout or logs. No pseudo terminal is requested.
"""
import argparse
import os
from pathlib import Path
import subprocess

WORKSPACE = Path(__file__).resolve().parents[6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command')
    parser.add_argument('--env-file',type=Path,default=WORKSPACE/'.env')
    parser.add_argument('--sudo',action='store_true')
    args = parser.parse_args()
    password = os.environ.get('PYNQ_PASSWORD')
    if password is None and args.env_file.is_file():
        for line in args.env_file.read_text(encoding='utf-8-sig').splitlines():
            key,sep,value = line.partition('=')
            if sep and key.strip() == 'PYNQ_PASSWORD':
                password = value.strip().strip('\"\'')
                break
    if args.sudo and not password:
        raise RuntimeError('PYNQ_PASSWORD is missing from the environment/.env')
    command = ('sudo -S -p "" '+args.command) if args.sudo else args.command
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','pynq',command],
                            input=password+'\n' if args.sudo else None,text=True)
    return result.returncode


if __name__ == '__main__': raise SystemExit(main())
