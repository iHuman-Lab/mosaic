

import pyxdf
import pandas as pd
import yaml
from pathlib import Path

def inspect_xdf_file(xdf_path):
	"""
	Reads an XDF file and prints the column names and first 100 rows for each stream.
	Args:
		xdf_path (str or Path): Path to the XDF file.
	"""
	streams, _ = pyxdf.load_xdf(str(xdf_path))
	for i, stream in enumerate(streams):
		name = stream['info'].get('name', [''])[0]
		print(f"\nStream {i+1}: {name}")
		# Try to parse as eyetracker (float) or game (json) or generic
		try:
			# Try to parse as float data (eyetracker)
			data = pd.DataFrame(stream['time_series'])
			if 'time_stamps' in stream:
				data.insert(0, 'timestamp', stream['time_stamps'])
		except Exception:
			# Try to parse as JSON (game)
			rows = []
			for ts, v in zip(stream.get('time_stamps', []), stream.get('time_series', [])):
				try:
					sample = v[0] if isinstance(v, (list, tuple)) else v
					sample = pd.json.loads(sample)
					sample['timestamp'] = ts
					rows.append(sample)
				except Exception:
					continue
			data = pd.DataFrame(rows)
		print("Columns:", list(data.columns))
		print(data.head(100))


if __name__ == "__main__":
	config_path = Path(__file__).parent / "config.yaml"
	with open(config_path, "r") as f:
		config = yaml.safe_load(f)
	# Get file path from config (use first subject as example)
	subjects = config.get("subjects", [])
	if not subjects:
		raise ValueError("No subjects found in config.yaml")
	subject_id = subjects[0]["id"] if isinstance(subjects[0], dict) else subjects[0]
	file_template = config["xdf"]["file_template"]
	xdf_path = file_template.replace("{subject}", subject_id)
	xdf_path = Path(__file__).parent.parent / xdf_path
	print(f"Inspecting file: {xdf_path}")
	inspect_xdf_file(xdf_path)
