import yaml

file_path = 'zero/v1/config/zero_test.yaml'
with open(file_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
print(config)
