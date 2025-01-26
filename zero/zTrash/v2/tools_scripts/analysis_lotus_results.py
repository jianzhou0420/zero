import csv
import json

# Path to the JSONL file


def get_all_data():
    seed = range(0, 20)

    summary_data = dict()
    summary_data['all'] = dict()
    summary_data['all']['total'] = 0
    summary_data['all']['success'] = 0
    summary_data['all']['sr'] = 0

    first_time_flag = True
    for s in seed:
        file_path = f"/media/jian/ssd4t/preds/seed{s}/results.jsonl"

        # Open and load the file
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line.strip()))

        # Print the loaded data

        collected_data = dict()

        for item in data:  # retrieve the data
            if item['task'] not in collected_data.keys():
                collected_data[item['task']] = dict()
                collected_data[item['task']]['names'] = item['task']
                collected_data[item['task']]['total'] = item['num_demos']
                collected_data[item['task']]['success'] = round(item['num_demos'] * item['sr'])

            else:
                collected_data[item['task']]['total'] += item['num_demos']
                collected_data[item['task']]['success'] += round(item['num_demos'] * item['sr'])

        overall_total = 0
        overall_success = 0
        for task in collected_data:
            if first_time_flag:
                summary_data[task] = dict()
                summary_data[task]['names'] = task
                summary_data[task]['total'] = 0
                summary_data[task]['success'] = 0
            collected_data[task]['sr'] = collected_data[task]['success'] / collected_data[task]['total']
            overall_total += collected_data[task]['total']
            overall_success += collected_data[task]['success']
            summary_data[task]['total'] += collected_data[task]['total']
            summary_data[task]['success'] += collected_data[task]['success']

        overall_sr = overall_success / overall_total

        collected_data['overall'] = dict()
        collected_data['overall']['names'] = 'overall'
        collected_data['overall']['total'] = overall_total
        collected_data['overall']['success'] = overall_success
        collected_data['overall']['sr'] = overall_sr

        summary_data['all']['total'] += overall_total
        summary_data['all']['success'] += overall_success

        # save_path = f"/media/jian/ssd4t/preds/seed{s}/collected_results.jsonl"
        # with open(save_path, "w") as file:
        #     for entry in collected_data:
        #         json_line = json.dumps(collected_data[entry])
        #         file.write(json_line + "\n")
        first_time_flag = False

    for item in summary_data:
        if item == 'all':
            continue
        summary_data[item]['sr'] = summary_data[item]['success'] / summary_data[item]['total']
    summary_data['all']['sr'] = summary_data['all']['success'] / summary_data['all']['total']

    save_path = f"/media/jian/ssd4t/preds/summary_results.jsonl"
    with open(save_path, "w") as file:
        for entry in summary_data:
            json_line = json.dumps(summary_data[entry])
            file.write(json_line + "\n")

    return summary_data, save_path


def jsonl_to_csv(jsonl_file_path, csv_file_path):
    """
    Converts a JSONL file to a CSV file.

    Parameters:
    - jsonl_file_path: Path to the input JSONL file.
    - csv_file_path: Path to the output CSV file.
    """
    data = []
    fieldnames = set()

    # Read and parse the JSONL file
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
                fieldnames.update(json_obj.keys())
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                continue

    # Define the order of columns for CSV
    # You can customize the order as needed
    ordered_fieldnames = ['names', 'total', 'success', 'sr']

    # Ensure all required fields are present
    for field in ordered_fieldnames:
        if field not in fieldnames:
            fieldnames.add(field)

    # Write data to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ordered_fieldnames)

        # Write the header
        writer.writeheader()

        # Write each row, handling missing fields
        for entry in data:
            # Use dict.get to provide a default value for missing keys
            row = {field: entry.get(field, '') for field in ordered_fieldnames}
            writer.writerow(row)

    print(f"Successfully converted '{jsonl_file_path}' to '{csv_file_path}'.")


data, path = get_all_data()
jsonl_to_csv(path, "/media/jian/ssd4t/preds/summary_results.csv")
