import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, default='config.yaml', nargs='?')
args = parser.parse_args()

with open(args.config, 'r', encoding='utf-8') as f:
    tmp_data = f.read()
    config = yaml.load(tmp_data)

