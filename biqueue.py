
class Biqueue:
	def __init__(self):
		self.data = []

	def push_back(self, obj):
		self.data.append(obj)

	def pop_back(self):
		return self.data.pop()

	def push_front(self, obj):
		self.data = [obj] + self.data

	def pop_front(self):
		obj = self.data[0]
		self.data = self.data[1:]
		return obj

	def get(self, idx):
		return self.data[idx]
