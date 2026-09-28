from kaggle.api.kaggle_api_extended import KaggleApi


api = KaggleApi()
api.authenticate()
token = None
page = 0
prefix_counts = {}
samples = {}

while True:
    response = api.dataset_list_files(
        "lee735/thesis-dataset-new", page_token=token, page_size=200
    )
    page += 1
    for dataset_file in response.dataset_files:
        name = dataset_file.name
        checks = {
            "annotations": "/annotations_positions/",
            "camera_1": "/Image_subsets/C1/",
            "camera_7": "/Image_subsets/C7/",
            "intrinsic": "/calibrations/intrinsic_zero/",
            "extrinsic": "/calibrations/extrinsic/",
            "drop_45": "/drop_annotations/drop_45/annotations_positions/",
        }
        if name.startswith("Wildtrack/"):
            for label, needle in checks.items():
                if needle in name:
                    prefix_counts[label] = prefix_counts.get(label, 0) + 1
                    samples.setdefault(label, name)
    token = response.next_page_token
    if not token:
        break

print("pages:", page)
print("prefix counts:", prefix_counts)
print("sample paths:")
for label, name in samples.items():
    print(f"{label}: {name}")
