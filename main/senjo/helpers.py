import json
import numpy as np
class DefaultJSONEncoder(json.JSONEncoder):
	def default(self, obj, *args, **kwargs):
		if hasattr(obj, 'to_json'):
			return obj.to_json()
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		try:
			return json.JSONEncoder.default(self,obj)
		except TypeError:
			return obj.__str__()
