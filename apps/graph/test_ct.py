from circuit_tracer.replacement_model import ReplacementModel
import inspect
print(inspect.signature(ReplacementModel.get_activations))
print(inspect.signature(ReplacementModel.generate))
print(inspect.signature(ReplacementModel.__call__))
