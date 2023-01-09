import csv

def get_column_data(filename, column_name):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        column_data = []
        for row in reader:
            column_data.append(row[column_name])
    return column_data

